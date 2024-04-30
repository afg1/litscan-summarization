import os
from pathlib import Path

import click
import lmql
import polars as pl
import psycopg2
from tqdm import tqdm
from transformers import AutoTokenizer


def w_pbar(pbar, func):
    def foo(*args, **kwargs):
        pbar.update(1)
        return func(*args, **kwargs)

    return foo


def write_output(evaluated: dict[str, list], output: str) -> None:
    df = pl.DataFrame(evaluated)

    output_loc = Path(f"{output}")
    if output_loc.exists():
        existing = pl.read_parquet(output_loc)
        df = existing.vstack(df).unique("pmcid")
    df.write_parquet(output)


@lmql.query(decoder="sample", n=1, temperature=0.1, max_len=4096)
def classify_sentence(sentence, rna_id):
    '''lmql
    """
    <|user|>
    You are an academic with experience in molecular biology who has been
    asked to help identify interesting text snippets from papers about ncRNA.
    Below is a sentence from a paper. We need to filter out sentences that
    are not interesting to ncRNA scientists. You will briefly analyse the sentence
    before classifying it, giving your reasoning why the sentence is
    interesting to ncRNA scientists or not. Interesting sentences tell the reader
    something about the function, localisation, expression, or other biological
    aspects of an ncRNA. Uninteresting sentences might simply be a list of ncRNA
    identifiers, may use the ncRNA identifier in an equation, or may have the
    identifier within a list of primers. If you see DNA/RNA sequence in a sentence
    you should classify it as not interesting automatically. Sentences which are mainly
    composed of lists of ncRNA identifiers are not interesting.

    This sentence may be about the ncRNA '{rna_id}', and should mention it in the correct context
    ###
    {sentence}
    ###
    Is the sentence demarcated by ### interesting to ncRNA scientists? Does it provide useful information about {rna_id}? If so, state why in a single sentence.<|end|>
    <|assistant|>[ANALYSIS]
    """ where len(TOKENS(ANALYSIS)) < 256 STOPS_AT(ANALYSIS, "<|end|>")

    """Therefore, in the context of ncRNA this sentence is
    [CLS]""" distribution CLS in ["interesting", "not interesting"]

    '''


def classify_sentencess_df(sentence, rna_id, model):
    """
    Basically a wrapper function to apply the LLM classification across a
    dataframe
    """

    r = classify_sentence(sentence, rna_id, model=model)
    return r.variables["P(CLS)"][0][1]


QUERY = """select
lsr.pmcid, lsr.job_id, sentence, location

from litscan_result lsr
join litscan_database lsdb
	on lsdb.job_id = lsr.job_id
join litscan_body_sentence lsa
	on lsa.result_id = lsr.id
where lsdb.name = %s
and location <> 'other'

union

select
lsr.pmcid, lsr.job_id, sentence, 'abstract'
from litscan_result lsr
join litscan_database lsdb
	on lsdb.job_id = lsr.job_id
join litscan_abstract_sentence lsa
	on lsa.result_id = lsr.id
where lsdb.name = %s"""

PGDATABASE = os.getenv("PGDATABASE")


@click.command()
@click.argument("sentences")
@click.argument("output")
@click.option(
    "--model_path", default="/Users/agreen/LLMs/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
)
@click.option("--database", default="flybase")
@click.option("--ngl", default=1)
@click.option("--chunks", default=4)
@click.option("--gpu_number", default=0)
@click.option("--checkpoint_frequency", default=50)
def main(
    sentences,
    output,
    model_path,
    database,
    ngl,
    chunks,
    gpu_number,
    checkpoint_frequency,
):
    if sentences == "fetch":
        conn = psycopg2.connect(PGDATABASE)
        cur = conn.cursor()
        completed_query = cur.mogrify(
            QUERY,
            (
                database,
                database,
            ),
        )
        print(completed_query)
        sentences = (
            pl.read_database(completed_query, conn)
            .unique("pmcid")
            .with_row_count(name="index")
        )
        n_c = sentences.height // chunks
        pieces = [
            sentences.filter(pl.col("index").is_between(a * n_c, (a + 1) * n_c))
            for a in range(chunks)
        ]
        for n, piece in enumerate(pieces):
            piece.write_parquet(f"{output}_{n}.pq")
        exit()
    else:
        sentences = pl.read_parquet(sentences)

    ## Check for a previous checkpoint and resume where we left off
    if Path(output).exists():
        checkpointed = pl.read_parquet(output)
        sentences = sentences.join(checkpointed, on="pmcid", how="anti")
        print(f"Resuming from checkpoint, {sentences.height} to go")

    ## Set environment with supplied GPU ID
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_number)

    # abstract_text = """RNase P RNA (RPR), the catalytic subunit of the essential RNase P ribonucleoprotein, removes the 5' leader from precursor tRNAs. The ancestral eukaryotic RPR is a Pol III transcript generated with mature termini. In the branch of the arthropod lineage that led to the insects and crustaceans, however, a new allele arose in which RPR is embedded in an intron of a Pol II transcript and requires processing from intron sequences for maturation. We demonstrate here that the Drosophila intronic-RPR precursor is trimmed to the mature form by the ubiquitous nuclease Rat1/Xrn2 (5') and the RNA exosome (3'). Processing is regulated by a subset of RNase P proteins (Rpps) that protects the nascent RPR from degradation, the typical fate of excised introns. Our results indicate that the biogenesis of RPR in vivo entails interaction of Rpps with the nascent RNA to form the RNase P holoenzyme and suggests that a new pathway arose in arthropods by coopting ancient mechanisms common to processing of other noncoding RNAs."""
    print(sentences)

    model = lmql.model(
        f"local:llama.cpp:{model_path}",
        tokenizer="microsoft/Phi-3-mini-128k-instruct",
        n_gpu_layers=ngl,
        n_ctx=4096,
    )
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    ## filter out abstracts with zero or too many tokens
    sentences = sentences.with_columns(
        n_tokens=pl.col("sentence").apply(lambda x: len(tokenizer.encode(x)))
    )
    sentences = sentences.filter(pl.col("n_tokens").is_between(100, 3000))

    # r = classify_abstract(abstract_text, model=model,  output_writer=lmql.printing)
    # print(r)
    # exit()
    print(sentences)
    print(sentences.describe())
    print(sentences.select(pl.col("n_tokens").sum()))

    classified_pmcids = []
    relevant_probability = []
    for idx, paper in tqdm(
        enumerate(sentences.iter_rows(named=True)), total=sentences.height
    ):
        sentence = paper["sentence"]
        pmcid = paper["pmcid"]
        rna_id = paper["job_id"]

        rel_prob = classify_sentencess_df(sentence, rna_id, model)

        classified_pmcids.append(pmcid)
        relevant_probability.append(rel_prob)

        if idx % checkpoint_frequency == 0:
            evaluated = {
                "pmcid": classified_pmcids,
                "relevance_probability": relevant_probability,
            }
            write_output(evaluated, output)
            ## Reset accumulating lists
            classified_pmcids = []
            relevant_probability = []

    ## Finished, so write everything else
    evaluated = {
        "pmcid": classified_pmcids,
        "relevance_probability": relevant_probability,
    }
    write_output(evaluated, output)


if __name__ == "__main__":
    main()
