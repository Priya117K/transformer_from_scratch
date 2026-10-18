# transformer_from_scratch

## English–Italian translation app

Install dependencies and place the trained model checkpoint at
`transformer_en_it.pt` in the project root:

```powershell
pip install -r requirements.txt
python inference.py
```

The app listens on `0.0.0.0:7860` and also creates a Gradio share link. It
supports greedy or beam-search decoding, temperature and repetition-penalty
controls, and selectable decoder-layer/head cross-attention plots.

### Docker

Build the image from the repository root:

```powershell
docker build -t en-it-translator .
```

The checkpoint is intentionally excluded from the image. Mount it read-only
when starting the container:

```powershell
docker run --rm -p 7860:7860 `
  -v "${PWD}/transformer_en_it.pt:/app/transformer_en_it.pt:ro" `
  en-it-translator
```

The checkpoint path can be changed with `MODEL_PATH`. The English and Italian
tokenizer paths can be set with `TOKENIZER_EN_PATH` and `TOKENIZER_IT_PATH`.