# BDC Chatbot Streamlit Client

Streamlit client for the [BDC Chatbot](https://github.com/RENCI/BDC_Chatbot/).

## Copy environment variables
Copy `.env_example` to `.env` and make any necessary changes

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run chatbot

```bash
streamlit run client.py
```

or with more controls

```bash
streamlit run client.py --server.port=8501 --server.enableCORS=false --server.address=0.0.0.0
```

## Deployment

### Build Docker Image

```bash
docker build -t containers.renci.org/comms/bdcbot-client:0.1.0 .

```

TBD
