#!/usr/bin/env python3
import os, json, yaml, subprocess
import feedparser
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from diffusers import StableDiffusionPipeline
import torch
from datetime import datetime

# --- 1. Ayarları Oku ---
with open("config.yml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
feeds = [
    "https://hnrss.org/frontpage",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"
]
max_per_feed = 5

# --- 2. Fikirleri Topla ---
ideas = []
for url in feeds:
    d = feedparser.parse(url)
    for entry in d.entries[:max_per_feed]:
        ideas.append({"title": entry.title, "link": entry.link})

# --- 3. Metin ve Görsel Motorlarını Kur ---
# 3.1. Metin için GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model     = AutoModelForCausalLM.from_pretrained("gpt2")
gen       = pipeline("text-generation", model=model, tokenizer=tokenizer,
                    device_map="auto", temperature=0.7, max_length=512)
# 3.2. Görsel için Stable Diffusion
sd_model_id = "runwayml/stable-diffusion-v1-5"
device = "cuda" if torch.cuda.is_available() else "cpu"
sd = StableDiffusionPipeline.from_pretrained(
    sd_model_id,
    torch_dtype=torch.float16 if device=="cuda" else torch.float32
)
sd = sd.to(device)

# --- 4. İçerik & Görsel Üretimi (_posts formatı) ---
os.makedirs("_posts", exist_ok=True)
os.makedirs("images", exist_ok=True)

for idea in ideas:
    title = idea["title"]
    # 4.1. slug ve tarih
    date = datetime.utcnow().strftime("%Y-%m-%d")
    slug = "".join(c for c in title.lower() if c.isalnum() or c==" ").replace(" ", "-")
    md_path = f"_posts/{date}-{slug}.md"
    img_path = f"images/{slug}.png"

    # 4.2. Makale üret
    prompt = (
        f"Write a detailed blog post in Turkish about: '{title}'. "
        "Include intro, body and conclusion with at least three paragraphs."
    )
    text = gen(prompt)[0]["generated_text"]

    # 4.3. Affiliate bağlantıları
    aff = cfg.get("affiliate", {})
    footer = "\n---\n## Ürün Önerileri (Affiliate)\n"
    for plat, tag in aff.items():
        if plat.lower() == "amazon":
            url = f"https://www.amazon.com/dp/?tag={tag}"
        else:
            url = f"https://www.hepsiburada.com/?tag={tag}"
        footer += f"- [{plat.capitalize()}]({url})\n"

    # 4.4. Markdown dosyasını _posts/ klasörüne Front Matter’lı yaz
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write("layout: post\n")
        f.write(f"title: \"{title}\"\n")
        f.write(f"date: {date}T00:00:00+00:00\n")
        f.write("categories: ai pasif gelir\n")
        f.write("---\n\n")
        f.write(text + "\n\n" + footer)
    print(f"✅ Oluşturuldu: {md_path}")

    # 4.5. Görsel üret
    image = sd(title).images[0]
    image.save(img_path)
    print(f"🖼️ Oluşturuldu: {img_path}")

print("🎉 Tüm yazılar `_posts/` klasörüne hazırlandı.")
