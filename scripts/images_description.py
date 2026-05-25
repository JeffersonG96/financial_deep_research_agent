from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from pathlib import Path
import logging
import asyncio
import time

if load_dotenv():
    print("Cargado correctamente")

from PIL import Image

import io
import base64

IMG_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/data/procesed/images"
OUTPUT_DIR = "D:/Cursos/Agentes/financial_deep_research_agent/data/procesed/images_desc"
print(IMG_DIR)

# LLM
MODEL_NAME = "gemini-2.5-flash"
llm = ChatGoogleGenerativeAI(model=MODEL_NAME, temperature = 0)

system_prompt = """You are a senior financial analyst this financial specialized in reading and interpreting financial charts.

For charts and graphs:
- Identify the metric being measured
- List key data points and values
- Note significant trends (growth, decline, stability)

For tables:
- Extract column headers and key rows
- Note important values and totals

For text:
- Summarize key facts and numbers only
- Skip formatting, headers, and navigation elements

Be direct and factual. Focus on numbers, trends, and insights that would be useful for retrieval."""

def setup_logging(log_dir: Path, log_name: str = "images_description"):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"

    # Forzar reconfiguración
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a",encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return log_path

async def describe_images(out_dir: Path, image_path: Path):
    output_path = out_dir / image_path.parent.parent.name / image_path.parent.name / f"{image_path.stem}.md"
    if output_path.exists():
        logging.info("Ya existe, salto %s", image_path.stem)
        return
    
    image = Image.open(image_path)
    buffered = io.BytesIO()
    image.thumbnail((1536, 1536), Image.LANCZOS)

    image.save(buffered,format="PNG")
    img_bs64 = base64.b64encode(buffered.getvalue()).decode()

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
        content=[
            {'type': 'image_url', 'image_url':f"data:image/png;base64,{img_bs64}"},
            {'type': 'text', 'text': "Analyze this financial image"},
        ]
    )]

    response = await llm.ainvoke(messages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    logging.info("Image description: %s", image_path.name)
    setup_logging(Path(OUTPUT_DIR))

async def describe_all_images(batch_size=5):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    images = list(Path(IMG_DIR).rglob("page*.png"))
    logging.info("Imágenes encontradas: %d", len(images))
    print(len(images))

    for i in range(0, len(images), batch_size):
        batch = images[i: i + batch_size]
        tasks = [describe_images(Path(OUTPUT_DIR), img_path) for img_path in batch]
        await asyncio.gather(*tasks)
        logging.info("Batch %d - %d completado", i+1, i+len(batch))

async def main():
    start = time.time()
    await describe_all_images(batch_size=6)
    final = time.time() - start
    logging.info("Tiempo total: %.1f segundos (%.1f minutos)", final, final / 60)

if __name__ == "__main__":
    asyncio.run(main())
    