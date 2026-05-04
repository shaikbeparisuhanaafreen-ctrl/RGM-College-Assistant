from flask import Flask, render_template, request, redirect, session, jsonify
import csv
import os
import json
import whisper
from groq import Groq
from datetime import datetime
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect, DetectorFactory
from deep_translator import GoogleTranslator
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
app.secret_key = "supersecretkey"

# ================== LOAD WHISPER ==================
model = whisper.load_model("base")
# MongoDB connection (LOCAL — matches your Compass)
client = MongoClient("mongodb://localhost:27017/")
db = client["chatbot_credentials"]
users_collection = db["User_Details"]

bcrypt = Bcrypt(app)
# ================== GROQ CLIENT ==================
client = Groq(
    api_key="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"  
)

# ================== COLLEGE LOCATION ==================
COLLEGE_LAT = 15.505343
COLLEGE_LNG = 78.377162

import math

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # km

    lat1, lon1, lat2, lon2 = map(math.radians,
                                 [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + \
        math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2

    c = 2 * math.asin(math.sqrt(a))
    return R * c
# 🎯 RETRIEVAL TUNING

SIMILARITY_THRESHOLD = 0.42   # balanced mode
IMAGE_SIMILARITY_THRESHOLD = 0.35
MAX_CONTEXT_CHUNKS = 6

# 🚀 LOAD RAG COMPONENTS

print("🔄 Loading embedding model...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

print("📦 Loading FAISS index...")
index = faiss.read_index("faiss_index.bin")

with open("faiss_metadata.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

print(f"✅ Index loaded with {index.ntotal} vectors")


# 🌍 CACHE

response_cache = {}

# ================== SUGGESTION GENERATOR ==================

def generate_suggestions_llm(user_query, answer):
    prompt = f"""
You are an assistant for RGMCET College.

Generate 3 relevant follow-up questions a student might ask next.

STRICT RULES:
- Only about RGMCET
- Max 8 words each
- No numbering
- Return ONLY a JSON array of strings
- If unrelated to college, return []

User question: {user_query}
Assistant answer: {answer}
"""

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        text = resp.choices[0].message.content.strip()

        import json
        data = json.loads(text)

        if isinstance(data, list):
            return data[:4]

    except Exception as e:
        print("Suggestion error:", e)

    # ✅ safe fallback
    return [
        "Tell me about admissions",
        "What courses are offered?",
        "Is hostel available?"
    ]

# 🌍 LANGUAGE UTILITIES

DetectorFactory.seed = 0


def detect_language(text):
    try:
        lang = detect(text)
        if lang.startswith("te"):
            return "te"
        elif lang.startswith("hi"):
            return "hi"
        elif lang.startswith("kn"):
            return "kn"
        elif lang.startswith("en"):
            return "en"
        else:
            return "en"
    except:
        return "en"


def translate_to_english(text, source_lang):
    if source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception as e:
        print("Translate→EN error:", e)
        return text


def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception as e:
        print("Translate←EN error:", e)
        return text



# 🎯 INTENT DETECTOR

def detect_query_intent(query_en: str):
    q = query_en.lower()

    if any(word in q for word in ["faculty", "professor", "hod", "staff"]):
        return "faculty"

    if any(word in q for word in ["lab", "laboratory", "facilities"]):
        return "labs"

    if any(word in q for word in ["overview", "about", "department"]):
        return "overview"

    return "general"
# ================== NAVIGATION INTENT ==================
def is_navigation_query(query_en: str) -> bool:
    q = query_en.lower()

    nav_keywords = [
    "distance",
    "how far",
    "how to reach",
    "route",
    "directions",
    "from my location",
    "near me",
    "way to",
    "travel",
    "reach",
    "coming from",
    "i am from"
    ]

    return any(k in q for k in nav_keywords)


# HOME

@app.route("/", methods=["GET", "POST"])
def home():
    username = session.get("username")
    current_chat_id = session.get("current_chat_id")

    history_file = f"chat_history_{username}.json" if username else None

    # ---------------- LOAD CHATS ----------------
    if username and history_file and os.path.exists(history_file):
        with open(history_file, "r") as f:
            all_chats = json.load(f)
    else:
        all_chats = session.get("guest_chats", []) if not username else []

    
    # 🚀 HANDLE MESSAGE
    
    if request.method == "POST":
        query = request.form.get("query", "").strip()

        # 🌍 detect language
        user_lang = detect_language(query)
        session["user_lang"] = user_lang

        # 🌍 translate to English
        query_en = translate_to_english(query, user_lang)
        query_lower = query_en.lower()

        # 🎯 detect intent (FIXED)
        intent = detect_query_intent(query_en)
        department_filter, specialization_filter = detect_filters(query_en)

        retrieved_images = []
        retrieved_documents = []

       
        # CACHE
       
        if query_lower in response_cache:
            cached = response_cache[query_lower]
            answer = cached["answer"]
            retrieved_images = cached["images"]
            if is_navigation_query(query_en):
                suggestions = [
                    "Show distance from my location",
                    "Best route to RGMCET",
                    "Nearest bus to RGMCET",
                    "Open directions in maps"
                ]
            else:
                suggestions = generate_suggestions_llm(query_en, answer)

        else:
            
            # 🔍 SMART RAG SEARCH (FIXED)
            
            query_embedding = embed_model.encode(
                [query_en],
                normalize_embeddings=True
            )

            scores, indices = index.search(
                np.array(query_embedding).astype("float32"),
                k=10
            )

            retrieved_chunks = []
            retrieved_images = []
            retrieved_documents = []

            best_score = float(scores[0][0])

            # -------------------------------------------------
            # ❗ If top match is very weak → no context
            # -------------------------------------------------
            if best_score < SIMILARITY_THRESHOLD:
    # allow top 1 chunk anyway
                    context = ""
            else:
                for rank, idx in enumerate(indices[0]):
                    if idx == -1:
                        continue

                    score = float(scores[0][rank])

                    # 🚫 Skip weak chunks
                    if score < SIMILARITY_THRESHOLD:
                        continue

                    doc = documents[idx]
                    doc = documents[idx]
                    # 🎯 SPECIALIZATION FILTER (highest priority)
                    if specialization_filter:
                        if doc.get("specialization") != specialization_filter:
                            continue
                        elif department_filter:
                            if doc.get("department") != department_filter:
                                continue

                    # ===============================
                    # 🧠 BUILD CONTEXT
                    # ===============================
                    text = doc.get("content") or doc.get("text_for_embedding", "")
                    if text:
                        retrieved_chunks.append(text)

                    # ===============================
                    # 🖼️ SMART IMAGE GATING
                    # ===============================
                    if score >= IMAGE_SIMILARITY_THRESHOLD:

                        doc_type = (doc.get("subcategory") or "").lower()
                        allow_images = False

                        if intent == "faculty" and any(word in doc_type for word in ["faculty", "professor", "staff"]):
                            allow_images = True
                        elif intent == "labs" and "lab" in doc_type:
                            allow_images = True
                        elif intent in ["overview", "general"]:
                            allow_images = True

                        if allow_images:
                            for img in doc.get("images", []):
                                if isinstance(img, str):
                                    img = {
                                        "url": img,
                                        "name": "",
                                        "designation": "",
                                        "qualification": "",
                                        "faculty_id": "",
                                        "contact": "",
                                        "email": "",
                                    }
                                elif isinstance(img, dict):
                                    img.setdefault("url", "")
                                    img.setdefault("name", "")
                                    img.setdefault("designation", "")
                                    img.setdefault("qualification", "")
                                    img.setdefault("faculty_id", "")
                                    img.setdefault("contact", "")
                                    img.setdefault("email", "")
                                else:
                                    continue

                                if img.get("url"):
                                    retrieved_images.append(img)

                    
                    # 📄 DOCUMENTS
                    
                    retrieved_documents.extend(doc.get("documents", []))

                    if len(retrieved_chunks) >= MAX_CONTEXT_CHUNKS:
                        break

                
                # 🧹 REMOVE DUPLICATE IMAGES
                
                seen = set()
                unique_images = []

                for img in retrieved_images:
                    url = img.get("url")
                    if url and url not in seen:
                        seen.add(url)
                        unique_images.append(img)

                retrieved_images = unique_images

                
                # 🧠 FINAL CONTEXT
                
                context = "\n\n".join(retrieved_chunks)
            
            # 🧠 SYSTEM PROMPT (STRICT)
            
            system_prompt = """
You are RGMCET College AI Assistant.
GOAL:
Provide helpful, complete, and well-structured answers about RGMCET.

STYLE:
- Friendly and professional
- Speak like helpful college staff
- Write in clear paragraphs
- Use simple formatting when helpful

ANSWER RULES:
- Use ONLY the provided context.
- Do NOT invent facts.
If the answer is partially available, provide the available information.
Only say "I couldn't find that in RGMCET data." when absolutely nothing relevant exists.

SMARTNESS:
- If multiple details exist, summarize them clearly.
- If a person is mentioned, include their role if available.
- If contact info exists, present it neatly.
- Prefer complete answers over one-line replies.
- Use clean bullet points when listing people.
- Avoid excessive markdown symbols.
LANGUAGE:
- Always answer in English.
"""

            
            # 🤖 LLM CALL
            
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",
                     "content": f"Context:\n{context}\n\nQuestion: {query_en}"},
                ],
            )

            answer = response.choices[0].message.content
            answer = translate_from_english(answer, user_lang)
            # 🧠 SMART SUGGESTIONS
            if is_navigation_query(query_en):
                suggestions = [
                    "Show distance from my location",
                    "Best route to RGMCET",
                    "Nearest bus to RGMCET",
                    "Open directions in maps"
                ]
            else:
                suggestions = generate_suggestions_llm(query_en, answer)
            # cache
            response_cache[query_lower] = {
                "answer": answer,
                "images": retrieved_images,
            }

        
        # 💬 SAVE CHAT
        
        if not current_chat_id:
            current_chat_id = str(datetime.now().timestamp())
            session["current_chat_id"] = current_chat_id
            all_chats.append(
                {"id": current_chat_id, "title": query[:40], "messages": []}
            )

        for chat in all_chats:
            if chat["id"] == current_chat_id:
                chat["messages"].append(
                    {
                        "query": query,
                        "response": answer,
                        "images": retrieved_images[:4],
                        "documents": retrieved_documents[:3],
                        "suggestions": suggestions,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                )

        if username:
            with open(history_file, "w") as f:
                json.dump(all_chats, f, indent=4)
        else:
            session["guest_chats"] = all_chats

        return redirect("/")

    
    # LOAD CURRENT CHAT
    
    current_messages = []
    for chat in all_chats:
        if chat.get("id") == session.get("current_chat_id"):
            current_messages = chat.get("messages", [])
            break

    return render_template(
        "index.html",
        username=username,
        chats=all_chats,
        current_messages=current_messages,
    )

# LOGIN

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    username = request.form.get("username")
    password = request.form.get("password")

    user = users_collection.find_one({"username": username})

    if user and bcrypt.check_password_hash(user["password"], password):
        session["username"] = username
        return redirect("/")

    return "Invalid credentials"
# REGISTER

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template("register.html")

    username = request.form.get("username")
    email = request.form.get("email")
    mobile = request.form.get("mobile")
    password = request.form.get("password")

    # 🚫 check duplicate username
    if users_collection.find_one({"username": username}):
        return "Username already exists"

    # 🚫 check duplicate email
    if users_collection.find_one({"email": email}):
        return "Email already registered"

    # 🔐 hash password
    hashed_password = bcrypt.generate_password_hash(password).decode("utf-8")

    # 💾 insert user
    users_collection.insert_one({
        "username": username,
        "email": email,
        "mobile": mobile,
        "password": hashed_password,
        "created_at": datetime.utcnow()
    })

    session["username"] = username
    return redirect("/")

# LOGOUT

@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# NEW CHAT

@app.route("/new_chat")
def new_chat():
    session["current_chat_id"] = None
    return redirect("/")

# SWITCH CHAT

@app.route("/chat/<chat_id>")
def open_chat(chat_id):
    session["current_chat_id"] = chat_id
    return redirect("/")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio = request.files["audio"]
    audio.save("temp_audio.wav")
    result = model.transcribe("temp_audio.wav")
    return jsonify({"text": result["text"]})

@app.route("/get_distance", methods=["POST"])
def get_distance():
    data = request.json

    user_lat = float(data.get("lat"))
    user_lng = float(data.get("lng"))
    user_address = data.get("user_address", "Your current location")
    college_address = (
    "Rajeev Gandhi Memorial College of Engineering and Technology, "
    "Nerawada X Roads, Nandyal, Andhra Pradesh 518501, India"
    )

    distance = haversine_distance(
        user_lat, user_lng,
        COLLEGE_LAT, COLLEGE_LNG
    )

    # 🚦 smart classification
    if distance <= 5:
        transport = "near"
        suggestion = "You are very close. Auto, bike, or self transport is best."
        transport_details = [
        "🚶 Walking possible",
        "🏍 Bike or auto recommended",
        "🚗 Private vehicle fastest"
        ]
    elif distance <= 25:
        transport = "medium"
        suggestion = "Moderate distance. Direct bus or private vehicle recommended."
        transport_details = [
        "🚌 Look for direct APSRTC buses to Nandyal",
        "🚗 Car/bike gives faster travel",
        "📍 Reach Nandyal bus stand then local auto"
        ]
    else:
        transport = "far"
        suggestion = "Long distance. APSRTC/TSRTC or train recommended."
        transport_details = [
        "🚌 APSRTC/TSRTC buses to Nandyal available",
        "🚆 Nearest major railway station: Nandyal",
        "📍 From Nandyal take auto/bus to RGMCET"
        ]

    return jsonify({
        "distance_km": round(distance, 2),
    "transport_type": transport,
    "suggestion": suggestion,
    "transport_details": transport_details,
    "user_address": user_address,
    "college_address": college_address,
    "college_lat": COLLEGE_LAT,
    "college_lng": COLLEGE_LNG
    })
@app.route("/navigation")
def navigation():
    return render_template("navigation.html")
@app.route("/save_location", methods=["POST"])
def save_location():

    data = request.json
    print("LOCATION RECEIVED:", data)

    user_data = {
        "lat": data.get("lat"),
        "lng": data.get("lng")
    }

    with open("user_location.json", "w") as f:
        json.dump(user_data, f)

    print("Location saved!")

    return jsonify({"status":"saved"})
@app.route("/get_campus_locations")
def get_campus_locations():

    import json

    with open("campus_location.json","r") as f:
        data = json.load(f)

    return jsonify(data)
def detect_filters(query):
    q = query.lower()

    specialization = None
    department = None

    # SPECIALIZATION detection
    if "aiml" in q or "ai ml" in q:
        specialization = "aiml"
        department = "cse"

    elif "business systems" in q or "cse bs" in q:
        specialization = "business_systems"
        department = "cse"

    elif "data science" in q or "cse ds" in q:
        specialization = "data_science"
        department = "cse"

    elif "cyber security" in q or "cybersecurity" in q:
        specialization = "cyber_security"
        department = "cse"

    elif "cse" in q:
        department = "cse"

    return department, specialization
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, threaded=True)