import argparse
import io
import PIL
import duckdb
import lancedb
import lance
import numpy as np
import pyarrow.compute as pc
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizerFast
import gradio as gr

MODEL_ID = None
MODEL = None
TOKENIZER = None
PROCESSOR = None


def create_table(dataset):
    db = lancedb.connect("~/datasets/demo")
    if "diffusiondb" in db.table_names():
        return db.open_table("diffusiondb")

    data = lance.dataset(dataset).to_table()
    tbl = db.create_table(
        "diffusiondb", data.filter(~pc.field("prompt").is_null()), mode="overwrite"
    )
    # 新版 lancedb (>=0.10) 的 native FTS 一次只能为一个字段建索引
    tbl.create_fts_index("prompt")

    return tbl


def setup_clip_model(model_id):
    global MODEL_ID, MODEL, TOKENIZER, PROCESSOR
    MODEL_ID = model_id
    TOKENIZER = CLIPTokenizerFast.from_pretrained(MODEL_ID)
    MODEL = CLIPModel.from_pretrained(MODEL_ID)
    PROCESSOR = CLIPProcessor.from_pretrained(MODEL_ID)


def embed_func(query):
    inputs = TOKENIZER([query], padding=True, return_tensors="pt")
    text_features = MODEL.get_text_features(**inputs)
    emb = text_features.detach().numpy()[0]
    # CLIP 输出的 embedding 未归一化，必须先 L2 normalize 才能正确做 cosine 相似度检索
    emb = emb / np.linalg.norm(emb)
    return emb.astype("float32")


def find_image_vectors(query):
    emb = embed_func(query)
    code = (
        "import lancedb\n"
        "db = lancedb.connect('~/datasets/demo')\n"
        "tbl = db.open_table('diffusiondb')\n\n"
        f"embedding = embed_func('{query}')\n"
        "tbl.search(embedding).metric('cosine').limit(9).to_pandas()"
    )
    return (
        _extract(tbl.search(emb).metric("cosine").limit(9).to_pandas()),
        code,
    )


def find_image_keywords(query):
    code = (
        "import lancedb\n"
        "db = lancedb.connect('~/datasets/demo')\n"
        "tbl = db.open_table('diffusiondb')\n\n"
        f"tbl.search('{query}', query_type='fts').limit(9).to_pandas()"
    )
    return (
        _extract(tbl.search(query, query_type="fts").limit(9).to_pandas()),
        code,
    )


def find_image_sql(query):
    code = (
        "import lancedb\n"
        "import duckdb\n"
        "db = lancedb.connect('~/datasets/demo')\n"
        "tbl = db.open_table('diffusiondb')\n\n"
        "diffusiondb = tbl.to_lance()\n"
        f"duckdb.sql('{query}').to_df()"
    )
    diffusiondb = tbl.to_lance()
    return (_extract(duckdb.sql(query).to_df()), code)


def _extract(df):
    image_col = "image"
    return [
        (PIL.Image.open(io.BytesIO(row[image_col])), row["prompt"])
        for _, row in df.iterrows()
    ]

def create_gradio_dash(server_name="0.0.0.0", server_port=7860, share=True):
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Tab("Embeddings"):
                vector_query = gr.Textbox(
                    value="portraits of a person", show_label=False
                )
                b1 = gr.Button("Submit")
            with gr.Tab("Keywords"):
                keyword_query = gr.Textbox(value="ninja turtle", show_label=False)
                b2 = gr.Button("Submit")
            with gr.Tab("SQL"):
                sql_query = gr.Textbox(
                    value="SELECT * FROM diffusiondb WHERE cfg = 12 LIMIT 9",
                    show_label=False,
                )
                b3 = gr.Button("Submit")
        with gr.Row():
            code = gr.Code(label="Code", language="python")
        with gr.Row():
            gallery = gr.Gallery(
                label="Found images", show_label=False, elem_id="gallery"
            ).style(columns=[3], rows=[3], object_fit="contain", height="auto")

        b1.click(find_image_vectors, inputs=vector_query, outputs=[gallery, code])
        b2.click(find_image_keywords, inputs=keyword_query, outputs=[gallery, code])
        b3.click(find_image_sql, inputs=sql_query, outputs=[gallery, code])

    demo.launch(server_name=server_name, server_port=server_port, share=share)


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--dataset", type=str, default="rawdata.lance")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="绑定的网卡地址，0.0.0.0 表示监听全部网卡")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", default=True,
                        help="是否生成 gradio.live 公网临时链接")
    parser.add_argument("--no-share", dest="share", action="store_false")
    return parser.parse_args()

if __name__ == "__main__":
    args = args_parse()
    setup_clip_model(args.model_id)
    tbl = create_table(args.dataset)
    create_gradio_dash(server_name=args.host, server_port=args.port, share=args.share)
