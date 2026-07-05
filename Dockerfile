FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data outputs

EXPOSE 8501

# Default: run the pipeline once, then launch the dashboard.
CMD ["sh", "-c", "python src/run_pipeline.py --workflows 420 --seed 42 && streamlit run src/app.py --server.address=0.0.0.0 --server.port=8501"]
