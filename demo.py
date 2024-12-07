from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, pipeline
import random
import json
import os
from datetime import datetime
import subprocess

# Flask app initialization
app = Flask(__name__)
CORS(app)

# Default model initialization (starting with base model)
model_name = "facebook/blenderbot-400M-distill"  # Default BlenderBot model
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)

# Sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")


class MentalHealthDiary:
    def __init__(self):
        self.user_info = {}  # Store user information (name and age)
        self.conversation_log = []  # Store conversation logs
        self.logs_dir = os.path.join(os.path.dirname(__file__), 'conversation_logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        self.model_finetuned = False  # Track if Fine-Tuned model is available

    def analyze_sentiment(self, user_input):
        """Analyze the sentiment of the user input."""
        result = sentiment_analyzer(user_input)
        return result[0]["label"]  # "POSITIVE" or "NEGATIVE"

    def generate_response(self, sentiment, user_input):
        """Generate a response combining predefined responses and model-generated responses."""
        # Handle specific questions directly
        if user_input.lower() in ["who are you?", "who r u?"]:
            return "I'm Penko, your friendly mental health companion."

        # Short or vague responses
        if len(user_input.split()) < 3:
            if sentiment == "NEGATIVE":
                return "I hear you. Sometimes it's hard to put feelings into words. I'm here for you."
            return "That's okay. Whenever you're ready to share more, I'm here."

        # Use Blenderbot for dynamic response generation
        inputs = tokenizer(user_input, return_tensors="pt")
        reply_ids = model.generate(**inputs)
        model_response = tokenizer.decode(reply_ids[0], skip_special_tokens=True)

        # Sentiment-based empathy
        if sentiment == "POSITIVE":
            empathic_response = random.choice([
                "That's wonderful to hear! What else made your day great?",
                "I'm so happy for you! Could you share more?",
            ])
        elif sentiment == "NEGATIVE":
            empathic_response = random.choice([
                "I'm sorry to hear that. Want to talk about it?",
                "It sounds tough. I'm here to listen anytime you need.",
            ])
        else:
            empathic_response = "I'm here to listen. Feel free to share more."

        # Blend empathy and model response for more natural flow
        if sentiment in ["POSITIVE", "NEGATIVE"]:
            return f"{empathic_response} {model_response}"
        return model_response

    def save_conversation(self):
        """Save the entire conversation log to a file."""
        if not self.conversation_log:
            return "No conversation to save."

        today = datetime.now().strftime("%Y-%m-%d")
        log_file = os.path.join(self.logs_dir, f"{today}_diary.json")
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_log, f, ensure_ascii=False, indent=4)
            self.conversation_log = []  # Clear the log after saving
            return f"Conversation saved to {log_file}"
        except Exception as e:
            return f"Error saving conversation: {e}"

    def retrain_model(self):
        print("Starting model retraining...")
        try:
            # retrain_blenderbot.py 스크립트 실행
            subprocess.run(["python", "retrain_blenderbot.py"], check=True)
            print("Model retrained successfully!")

            # 재학습된 모델 다시 로드
            global tokenizer, model
            tokenizer = BlenderbotTokenizer.from_pretrained("blenderbot-finetuned")
            model = BlenderbotForConditionalGeneration.from_pretrained("blenderbot-finetuned")
            self.model_finetuned = True  # Mark as fine-tuned
            print("Fine-Tuned model reloaded successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error during retraining: {e}")

    def process_user_input(self, user_input):
        sentiment = self.analyze_sentiment(user_input)

        if user_input.lower() == "exit":
            return self.save_conversation()

        ai_response = self.generate_response(sentiment, user_input)
        self.conversation_log.append({"timestamp": datetime.now().isoformat(),
                                       "user_input": user_input,
                                       "ai_response": ai_response})

        # Retrain model after 5 interactions
        if len(self.conversation_log) >= 5 and not self.model_finetuned:
            self.retrain_model()

        return ai_response


mental_health_diary = MentalHealthDiary()


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        if not mental_health_diary.user_info:
            return jsonify({"reply": "Hello! I'm PENGCO. What's your name and age?"})
        else:
            return jsonify({"reply": "How was your day today? Feel free to share your thoughts."})

    if user_input.lower() == "reset":
        mental_health_diary.user_info = {}
        return jsonify({"reply": "Your profile has been reset. What's your name and age?"})

    if not mental_health_diary.user_info:
        try:
            name, age = user_input.split(" ")
            mental_health_diary.user_info = {"name": name.capitalize(), "age": int(age)}
            return jsonify({"reply": f"Thank you {name.capitalize()}. You're {age} years old, right? If this is incorrect, type 'reset'. If it's correct, type 'yes'."})
        except ValueError:
            return jsonify({"reply": "Please provide your name and age in the format: 'Name Age'."})

    if user_input.lower() == "yes":
        name = mental_health_diary.user_info.get("name")
        return jsonify({"reply": f"Great to meet you, {name}! How was your day today? Feel free to share."})

    response = mental_health_diary.process_user_input(user_input)
    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
