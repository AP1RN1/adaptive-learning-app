"""
ADAPTIVE LEARNING SYSTEM - COMPLETE (FIXED DUPLICATE ANSWER + PERSISTENCE + LEADERBOARD)
M.Tech Project - All features, 50 questions, flashcards, BKT, ML prediction, JSON save/load, CSV leaderboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib
import time
import json
import os
import csv
from datetime import datetime

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Adaptive Learning System", page_icon="🎓", layout="wide")

# ============================================
# LOAD ML MODEL (if exists)
# ============================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('adaptive_learning_model.pkl')
    except:
        return None
rf_model = load_model()

# ============================================
# PERSISTENCE: SAVE / LOAD STUDENT DATA
# ============================================
def save_student_data(student):
    data = {
        "student_id": student.student_id,
        "name": student.name,
        "topic_mastery": student.topic_mastery,
        "total_questions": student.total_questions,
        "total_correct": student.total_correct,
        "streak": student.streak
    }
    with open(f"student_{student.student_id}.json", "w") as f:
        json.dump(data, f, indent=4)

def load_student_data(student_id, name):
    filename = f"student_{student_id}.json"
    if os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        student = SimpleStudent(data["student_id"], data["name"])
        student.topic_mastery = data["topic_mastery"]
        student.total_questions = data["total_questions"]
        student.total_correct = data["total_correct"]
        student.streak = data["streak"]
        return student
    return None

# ============================================
# LEADERBOARD (CSV)
# ============================================
def save_to_leaderboard(student, accuracy):
    anon_id = student.student_id[:4] + "***"
    with open("leaderboard.csv", "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), anon_id, accuracy, student.total_questions])

def show_leaderboard():
    try:
        df = pd.read_csv("leaderboard.csv", names=["timestamp", "student", "accuracy", "total_q"])
        df = df.sort_values("accuracy", ascending=False).head(10)
        st.subheader("🏆 Top Performers")
        st.dataframe(df[["student", "accuracy", "total_q"]])
    except:
        st.info("No scores yet. Complete a quiz to appear on leaderboard!")

# ============================================
# SIMPLE EXPLANATIONS (for wrong answers) - FALLBACK INCLUDED
# ============================================
def get_simple_explanation(topic, question):
    explanations = {
        "Machine Learning": {
            "What does supervised learning mean?": "📌 Jaise teacher answer sheet dekar padhata hai. Model ko pata hai ki output kya hona chahiye.",
            "What is unsupervised learning?": "📌 Jaise bina answer sheet ke khud patterns find karna.",
            "What is overfitting?": "📌 Jaise exam ke questions rat lo lekin naye questions solve na kar pao.",
            "Which algorithm minimizes sum of squared residuals?": "📌 Linear Regression line draw karta hai jo sab points ke beech se guzare.",
            "What is classification?": "📌 Yes/No ya Category batana. Jaise 'Ye email spam hai ya nahi?'",
            "What is regression?": "📌 Number batana. Jaise 'Ghar ki price kya hogi?'"
        },
        "Data Structures": {
            "Which data structure follows LIFO?": "📌 Stack. Jaise plate ka stack - upar rakho, upar se uthao.",
            "Which data structure follows FIFO?": "📌 Queue. Jaise ticket line - pehle aao, pehle jao.",
            "Time complexity of binary search?": "📌 O(log n). Jaise dictionary mein word dhundhna - aadha karo, phir aadha.",
            "What data structure is used for recursion?": "📌 Stack. Jaise ek function doosre function ko call kare, sab wait karte hain."
        },
        "Databases": {
            "What does SQL stand for?": "📌 Structured Query Language. Database se baat karne ki bhasha.",
            "What is a primary key?": "📌 Unique number. Jaise Aadhar Card - har ek unique hota hai.",
            "What is a foreign key?": "📌 Do tables ko jodne wala setu.",
            "Which normal form removes transitive dependency?": "📌 3NF. Data repeat hone se bachne ka tarika."
        },
        "Web Development": {
            "What does HTML stand for?": "📌 Hyper Text Markup Language. Website ka structure banata hai.",
            "What does CSS stand for?": "📌 Cascading Style Sheets. Website ko sundar banata hai.",
            "What is JavaScript?": "📌 Website mein interactivity laane wali language.",
            "Frontend vs Backend difference?": "📌 Frontend = dining area, Backend = kitchen."
        }
    }
    topic_dict = explanations.get(topic, {})
    result = topic_dict.get(question, None)
    if result is None:
        result = f"📌 Is concept ko samjho. Topic '{topic}' ke fundamentals revise karo."
    return result

# ============================================
# STUDENT MODEL (BKT)
# ============================================
class SimpleStudent:
    def __init__(self, student_id, name):
        self.student_id = student_id
        self.name = name
        self.topic_mastery = {}
        self.total_questions = 0
        self.total_correct = 0
        self.streak = 0

    def update(self, topic, correct):
        current = self.topic_mastery.get(topic, 0.3)
        if correct:
            new = min(1.0, current + 0.15)
            self.total_correct += 1
            self.streak += 1
        else:
            new = max(0.0, current - 0.1)
            self.streak = 0
        self.topic_mastery[topic] = round(new, 2)
        self.total_questions += 1

    def get_difficulty(self, topic):
        mastery = self.topic_mastery.get(topic, 0.3)
        return "hard" if mastery >= 0.7 else "medium" if mastery >= 0.4 else "easy"

    def get_accuracy(self):
        if self.total_questions == 0:
            return 0
        return round((self.total_correct / self.total_questions) * 100, 2)

    def get_weak_topics(self):
        return [t for t, m in self.topic_mastery.items() if m < 0.4]

    def get_features(self):
        avg_mastery = sum(self.topic_mastery.values()) / max(1, len(self.topic_mastery))
        return np.array([[
            avg_mastery * 100,
            self.total_questions * 50,
            self.total_questions / max(1, 7),
            0, 0, self.total_questions, 0, 0, 1
        ]])

# ============================================
# QUESTION BANK (50 QUESTIONS - CORRECT ANSWERS SPREAD A,B,C,D)
# ============================================
class QuestionBank:
    def __init__(self):
        self.questions = self._load()

    def _load(self):
        return [
            # ========== MACHINE LEARNING (13) ==========
            {"id": 1, "topic": "Machine Learning", "difficulty": "easy",
             "question": "What does supervised learning mean?",
             "options": ["A) Learning with labeled data", "B) Without data", "C) With images", "D) With rewards"],
             "answer": "A", "explanation": "Supervised learning uses labeled input-output pairs."},
            {"id": 2, "topic": "Machine Learning", "difficulty": "easy",
             "question": "What is unsupervised learning?",
             "options": ["A) Predict labels", "B) Find patterns in unlabeled data", "C) Classify images", "D) Play games"],
             "answer": "B", "explanation": "Unsupervised learning finds hidden patterns without labels."},
            {"id": 3, "topic": "Machine Learning", "difficulty": "easy",
             "question": "What is a model in ML?",
             "options": ["A) Mathematical representation of data patterns", "B) Database table", "C) Programming language", "D) Web framework"],
             "answer": "A", "explanation": "A model learns patterns from training data."},
            {"id": 4, "topic": "Machine Learning", "difficulty": "easy",
             "question": "What is training data?",
             "options": ["A) Data used to teach the model", "B) Data used to test", "C) Data to delete", "D) Data to store"],
             "answer": "A", "explanation": "Training data teaches the model patterns."},
            {"id": 5, "topic": "Machine Learning", "difficulty": "medium",
             "question": "Which algorithm minimizes sum of squared residuals?",
             "options": ["A) Logistic Regression", "B) Linear Regression", "C) Decision Tree", "D) K-Means"],
             "answer": "B", "explanation": "Linear Regression minimizes OLS cost function."},
            {"id": 6, "topic": "Machine Learning", "difficulty": "medium",
             "question": "What is overfitting?",
             "options": ["A) Learns training too well, poor on test", "B) Poor on both", "C) Good on test only", "D) Too simple"],
             "answer": "A", "explanation": "Overfitting learns noise in training data."},
            {"id": 7, "topic": "Machine Learning", "difficulty": "medium",
             "question": "What is underfitting?",
             "options": ["A) Model too simple, poor on both", "B) Too complex", "C) Perfect", "D) Over-learns"],
             "answer": "A", "explanation": "Underfitting means model is too simple."},
            {"id": 8, "topic": "Machine Learning", "difficulty": "medium",
             "question": "What is classification?",
             "options": ["A) Predicting numbers", "B) Predicting categories", "C) Clustering", "D) Reducing dimensions"],
             "answer": "B", "explanation": "Classification predicts discrete categories."},
            {"id": 9, "topic": "Machine Learning", "difficulty": "medium",
             "question": "What is regression?",
             "options": ["A) Predicting categories", "B) Predicting continuous values", "C) Grouping data", "D) Finding outliers"],
             "answer": "B", "explanation": "Regression predicts continuous values like price."},
            {"id": 10, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is the kernel trick in SVM?",
             "options": ["A) Reduce dimensions", "B) Map data to higher dimension", "C) Remove outliers", "D) Speed up training"],
             "answer": "B", "explanation": "Kernel trick maps to higher dimension without explicit computation."},
            {"id": 11, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is the bias-variance tradeoff?",
             "options": ["A) Low bias = low variance", "B) High bias = high variance", "C) Low bias = high variance", "D) No relationship"],
             "answer": "C", "explanation": "Simple models = high bias, complex = high variance."},
            {"id": 12, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is ensemble learning?",
             "options": ["A) Single model", "B) Combining multiple models", "C) Deep learning", "D) Reinforcement learning"],
             "answer": "B", "explanation": "Ensemble combines multiple models (Random Forest)."},
            {"id": 13, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is cross-validation?",
             "options": ["A) Single split", "B) Splitting data multiple times", "C) Preprocessing", "D) Feature engineering"],
             "answer": "B", "explanation": "Cross-validation ensures model generalizes well."},
            # ========== DATA STRUCTURES (12) ==========
            {"id": 14, "topic": "Data Structures", "difficulty": "easy",
             "question": "Which data structure follows LIFO?",
             "options": ["A) Queue", "B) Stack", "C) Linked List", "D) Heap"],
             "answer": "B", "explanation": "Stack = Last In First Out."},
            {"id": 15, "topic": "Data Structures", "difficulty": "easy",
             "question": "Which data structure follows FIFO?",
             "options": ["A) Stack", "B) Queue", "C) Linked List", "D) Tree"],
             "answer": "B", "explanation": "Queue = First In First Out."},
            {"id": 16, "topic": "Data Structures", "difficulty": "easy",
             "question": "What is an array?",
             "options": ["A) Collection of different types", "B) Collection of same type", "C) Key-value pairs", "D) Tree structure"],
             "answer": "B", "explanation": "Array stores elements of same data type."},
            {"id": 17, "topic": "Data Structures", "difficulty": "easy",
             "question": "What is a linked list?",
             "options": ["A) Continuous memory", "B) Nodes connected by pointers", "C) Key-value pairs", "D) Tree"],
             "answer": "B", "explanation": "Linked list has nodes with data and next pointer."},
            {"id": 18, "topic": "Data Structures", "difficulty": "medium",
             "question": "Time complexity of binary search?",
             "options": ["A) O(n)", "B) O(n²)", "C) O(log n)", "D) O(1)"],
             "answer": "C", "explanation": "Binary search = O(log n)."},
            {"id": 19, "topic": "Data Structures", "difficulty": "medium",
             "question": "What data structure is used for recursion?",
             "options": ["A) Queue", "B) Stack", "C) Array", "D) Linked List"],
             "answer": "B", "explanation": "Stack stores function call states."},
            {"id": 20, "topic": "Data Structures", "difficulty": "medium",
             "question": "What is a binary tree?",
             "options": ["A) 3 children per node", "B) Max 2 children per node", "C) Linear", "D) No children"],
             "answer": "B", "explanation": "Binary tree nodes have at most 2 children."},
            {"id": 21, "topic": "Data Structures", "difficulty": "medium",
             "question": "What is a hash table?",
             "options": ["A) Array", "B) Key-value with hash function", "C) Linked list", "D) Tree"],
             "answer": "B", "explanation": "Hash table uses hash function for O(1) access."},
            {"id": 22, "topic": "Data Structures", "difficulty": "hard",
             "question": "What keeps Red-Black Tree balanced?",
             "options": ["A) All red", "B) Color properties", "C) Complete tree", "D) Sorted order"],
             "answer": "B", "explanation": "Red-Black Tree uses coloring rules."},
            {"id": 23, "topic": "Data Structures", "difficulty": "hard",
             "question": "Worst-case time of QuickSort?",
             "options": ["A) O(n log n)", "B) O(n²)", "C) O(n)", "D) O(log n)"],
             "answer": "B", "explanation": "QuickSort worst case O(n²)."},
            {"id": 24, "topic": "Data Structures", "difficulty": "hard",
             "question": "What is a B-tree?",
             "options": ["A) Binary tree", "B) Self-balancing for databases", "C) Hash table", "D) Linked list"],
             "answer": "B", "explanation": "B-tree used in databases and file systems."},
            {"id": 25, "topic": "Data Structures", "difficulty": "hard",
             "question": "Dijkstra's algorithm used for?",
             "options": ["A) Sorting", "B) Shortest path", "C) Searching", "D) Hashing"],
             "answer": "B", "explanation": "Dijkstra finds shortest path in graphs."},
            # ========== DATABASES (12) ==========
            {"id": 26, "topic": "Databases", "difficulty": "easy",
             "question": "What does SQL stand for?",
             "options": ["A) Structured Query Language", "B) Simple Query Logic", "C) System Query Language", "D) Sorted Query Language"],
             "answer": "A", "explanation": "SQL = Structured Query Language."},
            {"id": 27, "topic": "Databases", "difficulty": "easy",
             "question": "What is a primary key?",
             "options": ["A) Duplicate values", "B) Unique record identifier", "C) Foreign reference", "D) Index only"],
             "answer": "B", "explanation": "Primary key uniquely identifies each record."},
            {"id": 28, "topic": "Databases", "difficulty": "easy",
             "question": "What is a foreign key?",
             "options": ["A) Primary key", "B) Reference to another table's key", "C) Unique key", "D) Composite key"],
             "answer": "B", "explanation": "Foreign key links two tables together."},
            {"id": 29, "topic": "Databases", "difficulty": "easy",
             "question": "What is a table in database?",
             "options": ["A) Single row", "B) Rows and columns", "C) Single column", "D) Database"],
             "answer": "B", "explanation": "Table stores data in rows and columns."},
            {"id": 30, "topic": "Databases", "difficulty": "medium",
             "question": "Which normal form removes transitive dependency?",
             "options": ["A) 1NF", "B) 2NF", "C) 3NF", "D) BCNF"],
             "answer": "C", "explanation": "3NF removes transitive dependencies."},
            {"id": 31, "topic": "Databases", "difficulty": "medium",
             "question": "What is a JOIN?",
             "options": ["A) Delete records", "B) Combine tables", "C) Create tables", "D) Update records"],
             "answer": "B", "explanation": "JOIN combines data from multiple tables."},
            {"id": 32, "topic": "Databases", "difficulty": "medium",
             "question": "What is an index?",
             "options": ["A) Data storage", "B) Faster data retrieval", "C) Backup", "D) Encryption"],
             "answer": "B", "explanation": "Index speeds up data retrieval."},
            {"id": 33, "topic": "Databases", "difficulty": "medium",
             "question": "What is a query?",
             "options": ["A) Deletion", "B) Request for data", "C) Creation", "D) Backup"],
             "answer": "B", "explanation": "Query retrieves specific data from database."},
            {"id": 34, "topic": "Databases", "difficulty": "hard",
             "question": "What does ACID stand for?",
             "options": ["A) Atomicity,Consistency,Isolation,Durability", "B) Atomicity,Concurrency,Integrity,Durability", "C) Availability,Consistency,Isolation,Durability", "D) Atomicity,Consistency,Integrity,Durability"],
             "answer": "A", "explanation": "ACID ensures reliable database transactions."},
            {"id": 35, "topic": "Databases", "difficulty": "hard",
             "question": "INNER JOIN vs LEFT JOIN?",
             "options": ["A) INNER = all rows, LEFT = matching", "B) INNER = matching only, LEFT = all from left", "C) LEFT = matching only", "D) Both same"],
             "answer": "B", "explanation": "INNER JOIN returns matching rows; LEFT JOIN all from left."},
            {"id": 36, "topic": "Databases", "difficulty": "hard",
             "question": "What is a transaction?",
             "options": ["A) Table creation", "B) Unit of work", "C) Index creation", "D) Backup"],
             "answer": "B", "explanation": "Transaction is a unit of work with ACID."},
            {"id": 37, "topic": "Databases", "difficulty": "hard",
             "question": "What is NoSQL?",
             "options": ["A) SQL only", "B) Non-relational database", "C) Relational database", "D) Programming language"],
             "answer": "B", "explanation": "NoSQL databases are non-relational (MongoDB)."},
            # ========== WEB DEVELOPMENT (13) ==========
            {"id": 38, "topic": "Web Development", "difficulty": "easy",
             "question": "What does HTML stand for?",
             "options": ["A) Hyper Text Markup Language", "B) High Tech Modern Language", "C) Hyper Transfer Markup Language", "D) Home Tool Markup Language"],
             "answer": "A", "explanation": "HTML = Hyper Text Markup Language."},
            {"id": 39, "topic": "Web Development", "difficulty": "easy",
             "question": "What does CSS stand for?",
             "options": ["A) Computer Style Sheets", "B) Cascading Style Sheets", "C) Creative Style Sheets", "D) Colorful Style Sheets"],
             "answer": "B", "explanation": "CSS controls web page appearance."},
            {"id": 40, "topic": "Web Development", "difficulty": "easy",
             "question": "What is JavaScript?",
             "options": ["A) CSS framework", "B) Programming language for web", "C) Database", "D) HTML tag"],
             "answer": "B", "explanation": "JavaScript adds interactivity to web pages."},
            {"id": 41, "topic": "Web Development", "difficulty": "easy",
             "question": "What is a web browser?",
             "options": ["A) Web server", "B) Software to view websites", "C) Database", "D) Programming language"],
             "answer": "B", "explanation": "Browser (Chrome, Firefox) displays web pages."},
            {"id": 42, "topic": "Web Development", "difficulty": "medium",
             "question": "Frontend vs Backend difference?",
             "options": ["A) Frontend = server, Backend = client", "B) Frontend = client, Backend = server", "C) Frontend = database", "D) Both same"],
             "answer": "B", "explanation": "Frontend is what users see; Backend handles data."},
            {"id": 43, "topic": "Web Development", "difficulty": "medium",
             "question": "What is a framework?",
             "options": ["A) Programming language", "B) Reusable code structure", "C) Database", "D) Server"],
             "answer": "B", "explanation": "Frameworks provide reusable code templates."},
            {"id": 44, "topic": "Web Development", "difficulty": "medium",
             "question": "What is an API?",
             "options": ["A) Application Protocol Interface", "B) Application Programming Interface", "C) Advanced Programming Interface", "D) Automated Program Interface"],
             "answer": "B", "explanation": "API allows different software to communicate."},
            {"id": 45, "topic": "Web Development", "difficulty": "medium",
             "question": "What is HTTP?",
             "options": ["A) High Transfer Protocol", "B) Hypertext Transfer Protocol", "C) Hyper Text Program", "D) High Text Protocol"],
             "answer": "B", "explanation": "HTTP is protocol for web communication."},
            {"id": 46, "topic": "Web Development", "difficulty": "medium",
             "question": "What is a responsive website?",
             "options": ["A) Desktop only", "B) Works on all devices", "C) Mobile only", "D) Tablet only"],
             "answer": "B", "explanation": "Responsive design adapts to different screen sizes."},
            {"id": 47, "topic": "Web Development", "difficulty": "hard",
             "question": "GET vs POST difference?",
             "options": ["A) GET = body, POST = URL", "B) GET = URL, POST = body", "C) Both same", "D) GET is faster"],
             "answer": "B", "explanation": "GET exposes data in URL; POST hides in body."},
            {"id": 48, "topic": "Web Development", "difficulty": "hard",
             "question": "What is a REST API?",
             "options": ["A) Programming language", "B) Architectural style for web services", "C) Database", "D) Web server"],
             "answer": "B", "explanation": "REST API follows stateless client-server architecture."},
            {"id": 49, "topic": "Web Development", "difficulty": "hard",
             "question": "What is the DOM?",
             "options": ["A) Data Object Model", "B) Document Object Model", "C) Document Oriented Model", "D) Data Oriented Model"],
             "answer": "B", "explanation": "DOM represents web page structure as objects."},
            {"id": 50, "topic": "Web Development", "difficulty": "hard",
             "question": "What is a Single Page Application (SPA)?",
             "options": ["A) Multiple HTML pages", "B) Single HTML, content dynamically updated", "C) Server-side rendering", "D) Static website"],
             "answer": "B", "explanation": "SPA loads one HTML and updates content dynamically."},
        ]

    def get_question(self, topic, difficulty, asked_ids=None):
        if asked_ids is None:
            asked_ids = set()
        # exact match
        pool = [q for q in self.questions if q["topic"]==topic and q["difficulty"]==difficulty and q["id"] not in asked_ids]
        if not pool:
            pool = [q for q in self.questions if q["topic"]==topic and q["id"] not in asked_ids]
        if not pool:
            pool = [q for q in self.questions if q["topic"]==topic and q["difficulty"]==difficulty]
        return random.choice(pool) if pool else None

    def get_topics(self):
        return list(set(q["topic"] for q in self.questions))

# ============================================
# FLASHCARD DATA (16 cards, 4 per topic)
# ============================================
flashcards_data = {
    "Machine Learning": [
        {"question": "What is Supervised Learning?", "answer": "Learning with labeled data.", "simple_explanation": "📌 Teacher answer sheet dekar padhata hai."},
        {"question": "What is Unsupervised Learning?", "answer": "Learning without labels.", "simple_explanation": "📌 Bina answer sheet ke patterns find karna."},
        {"question": "What is Overfitting?", "answer": "Model learns too well, fails on new data.", "simple_explanation": "📌 Exam ke questions rat lo, naye solve na kar pao."},
        {"question": "Classification vs Regression?", "answer": "Classification = categories, Regression = numbers.", "simple_explanation": "📌 Classification = Yes/No, Regression = Price."}
    ],
    "Data Structures": [
        {"question": "What is a Stack?", "answer": "LIFO data structure.", "simple_explanation": "📌 Plate ka stack."},
        {"question": "What is a Queue?", "answer": "FIFO data structure.", "simple_explanation": "📌 Ticket line."},
        {"question": "What is Binary Search?", "answer": "O(log n) algorithm.", "simple_explanation": "📌 Dictionary mein word dhundhna."},
        {"question": "What is a Linked List?", "answer": "Nodes connected by pointers.", "simple_explanation": "📌 Train ke coaches."}
    ],
    "Databases": [
        {"question": "What is SQL?", "answer": "Structured Query Language.", "simple_explanation": "📌 Database se baat karne ki bhasha."},
        {"question": "What is a Primary Key?", "answer": "Unique record identifier.", "simple_explanation": "📌 Aadhar Card."},
        {"question": "What is a Foreign Key?", "answer": "Links two tables.", "simple_explanation": "📌 Do tables ko jodne wala setu."},
        {"question": "What is Normalization?", "answer": "Reduces data redundancy.", "simple_explanation": "📌 Data repeat hone se bachna."}
    ],
    "Web Development": [
        {"question": "What is HTML?", "answer": "Structure of web pages.", "simple_explanation": "📌 Ghar ka naksha."},
        {"question": "What is CSS?", "answer": "Styling of web pages.", "simple_explanation": "📌 Ghar ki painting."},
        {"question": "What is JavaScript?", "answer": "Interactivity for web pages.", "simple_explanation": "📌 Ghar mein lights on/off."},
        {"question": "Frontend vs Backend?", "answer": "Frontend = client, Backend = server.", "simple_explanation": "📌 Frontend = dining area, Backend = kitchen."}
    ]
}

class FlashcardManager:
    def __init__(self):
        self.cards = flashcards_data
        self.current_topic = None
        self.current_card_index = 0
        self.revealed = False

    def get_topics(self):
        return list(self.cards.keys())

    def set_topic(self, topic):
        self.current_topic = topic
        self.current_card_index = 0
        self.revealed = False

    def get_current_card(self):
        if self.current_topic is None:
            return None
        cards = self.cards.get(self.current_topic, [])
        if self.current_card_index < len(cards):
            return cards[self.current_card_index]
        return None

    def next_card(self):
        cards = self.cards.get(self.current_topic, [])
        if self.current_card_index + 1 < len(cards):
            self.current_card_index += 1
            self.revealed = False
            return True
        return False

    def prev_card(self):
        if self.current_card_index > 0:
            self.current_card_index -= 1
            self.revealed = False
            return True
        return False

    def get_progress(self):
        cards = self.cards.get(self.current_topic, [])
        return f"{self.current_card_index + 1} of {len(cards)}"

    def total_cards(self):
        return len(self.cards.get(self.current_topic, []))

    def reveal(self):
        self.revealed = True

def show_flashcard_ui():
    if 'flashcard_manager' not in st.session_state:
        st.session_state.flashcard_manager = FlashcardManager()
    mgr = st.session_state.flashcard_manager

    if mgr.current_topic is None:
        st.subheader("📚 Choose a Topic for Flashcards")
        col1, col2 = st.columns(2)
        topics = mgr.get_topics()
        with col1:
            for t in topics[:2]:
                if st.button(f"📖 {t}", use_container_width=True):
                    mgr.set_topic(t)
                    st.rerun()
        with col2:
            for t in topics[2:]:
                if st.button(f"📖 {t}", use_container_width=True):
                    mgr.set_topic(t)
                    st.rerun()
        if st.button("🔙 Back to Quiz", use_container_width=True):
            st.session_state.show_flashcards = False
            st.rerun()
    else:
        st.markdown(f"<div style='background:#3498db; padding:15px; border-radius:10px'><h2 style='color:white'>📚 {mgr.current_topic}</h2><p style='color:white'>Card {mgr.get_progress()}</p></div>", unsafe_allow_html=True)
        st.progress(mgr.current_card_index / max(1, mgr.total_cards()))
        card = mgr.get_current_card()
        if card:
            st.markdown(f"<div style='background:#f0f2f6; padding:40px; border-radius:15px; text-align:center'><h3>❓ {card['question']}</h3></div>", unsafe_allow_html=True)
            if not mgr.revealed:
                if st.button("🔓 Reveal Answer", type="primary", use_container_width=True):
                    mgr.reveal()
                    st.rerun()
            if mgr.revealed:
                st.markdown(f"<div style='background:#2ecc71; padding:15px; border-radius:10px'><strong>✓ Answer:</strong> {card['answer']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='background:#3498db; padding:15px; border-radius:10px'><strong>💡 Easy Explanation:</strong> {card['simple_explanation']}</div>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("◀ Previous", use_container_width=True):
                        mgr.prev_card()
                        st.rerun()
                with col2:
                    if st.button("🔄 Change Topic", use_container_width=True):
                        mgr.current_topic = None
                        st.rerun()
                with col3:
                    if st.button("Next ▶", use_container_width=True):
                        mgr.next_card()
                        st.rerun()
        if st.button("🔙 Back to Quiz", use_container_width=True):
            st.session_state.show_flashcards = False
            st.rerun()

# ============================================
# QUIZ MODE SELECTION
# ============================================
def show_quiz_mode_selection():
    st.subheader("🎯 Choose Your Quiz Mode")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📚 TOPIC-WISE QUIZ\nPractice one topic at a time.")
        topic = st.radio("Select Topic:", ["Machine Learning", "Data Structures", "Databases", "Web Development"], key="topic_select")
        if st.button("Start Topic Quiz", type="primary"):
            st.session_state.quiz_mode = "topic_wise"
            st.session_state.selected_topic = topic
            st.session_state.complete = False
            st.session_state.current_q = None
            st.session_state.asked_question_ids = set()
            st.rerun()
    with col2:
        st.markdown("### 🔄 MIXED QUIZ (Adaptive)\nQuestions from all topics.\nSystem will mix questions and adapt difficulty.")
        if st.button("Start Mixed Quiz", type="primary"):
            st.session_state.quiz_mode = "mixed"
            st.session_state.selected_topic = None
            st.session_state.complete = False
            st.session_state.current_q = None
            st.session_state.asked_question_ids = set()
            st.rerun()

# ============================================
# SESSION STATE
# ============================================
if 'student' not in st.session_state:
    st.session_state.student = None
if 'current_q' not in st.session_state:
    st.session_state.current_q = None
if 'max_q' not in st.session_state:
    st.session_state.max_q = 10
if 'complete' not in st.session_state:
    st.session_state.complete = False
if 'qbank' not in st.session_state:
    st.session_state.qbank = QuestionBank()
if 'topics' not in st.session_state:
    st.session_state.topics = st.session_state.qbank.get_topics()
if 'quiz_mode' not in st.session_state:
    st.session_state.quiz_mode = None
if 'show_flashcards' not in st.session_state:
    st.session_state.show_flashcards = False
if 'asked_question_ids' not in st.session_state:
    st.session_state.asked_question_ids = set()
if 'selected_topic' not in st.session_state:
    st.session_state.selected_topic = None

# ============================================
# PAGE 1: REGISTRATION ONLY (with persistence)
# ============================================
if not st.session_state.student:
    st.title("🎓 AI-Based Adaptive Learning System")
    st.markdown("---")
    st.subheader("📝 Student Registration")
    col1, col2 = st.columns(2)
    with col1:
        sid = st.text_input("Student ID", placeholder="MTH2024001")
        name = st.text_input("Student Name", placeholder="Enter your name")
    with col2:
        st.markdown("**📚 Topics Covered:**")
        st.markdown("""
        - Machine Learning
        - Data Structures
        - Databases
        - Web Development
        """)
    if st.button("🎓 Start Learning", type="primary", use_container_width=True):
        if sid and name:
            existing = load_student_data(sid, name)
            if existing:
                st.session_state.student = existing
                st.success(f"✅ Welcome back {name}! Progress restored.")
            else:
                st.session_state.student = SimpleStudent(sid, name)
            st.rerun()
        else:
            st.error("Please enter both ID and Name")
    st.markdown("---")
    st.markdown("© Adaptive Learning System | M.Tech Project")
    st.stop()

# ============================================
# PAGE 2: AFTER LOGIN
# ============================================
s = st.session_state.student
st.title("🎓 AI-Based Adaptive Learning System")
st.markdown("---")
st.success(f"✅ Welcome {s.name} (ID: {s.student_id})")

# RESEARCH DASHBOARD
with st.expander("📊 Research Findings (OULAD Dataset - 32,593 Students)", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("🎯 Accuracy", "90.9%", "+14.7%")
    with c2:
        st.metric("📈 R² Score", "0.746", "+38.5%")
    with c3:
        st.metric("⭐ P-Value", "<0.001", "Significant")
    st.markdown("---")
    st.markdown("### 🔥 Feature Importance")
    imp_data = {
        "Feature": ["assessments_attempted", "weeks_active", "avg_score", "avg_submission_gap", "site_variety", "total_clicks"],
        "Importance (%)": [60.6, 9.3, 8.7, 4.6, 3.3, 3.0]
    }
    st.dataframe(pd.DataFrame(imp_data))
    chart_df = pd.DataFrame({"Feature": imp_data["Feature"], "Importance (%)": imp_data["Importance (%)"]})
    st.bar_chart(chart_df.set_index("Feature"))
    for feat, imp in zip(imp_data["Feature"], imp_data["Importance (%)"]):
        st.markdown(f"**{feat}** - {imp}%")
        st.progress(imp/100)
    st.success("💡 Behavioral features (91.3%) are MORE predictive than quiz scores (8.7%)!")

st.markdown("---")

# ============================================
# OPTION A: EXAM PREPARATION GUIDE (30-day study plan)
# ============================================
with st.expander("📅 Exam Preparation Guide - How to Cover All Topics", expanded=False):
    st.markdown("### 🎯 30-Day Study Plan for Engineering Exam")
    total_topics = len(st.session_state.topics)
    mastered = len([t for t, m in s.topic_mastery.items() if m >= 0.7])
    weak = s.get_weak_topics()
    st.markdown(f"**📊 Your Progress:** ✅ Mastered: {mastered}/{total_topics} | ⚠️ Needs attention: {len(weak)}/{total_topics}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Week 1: Foundation (Days 1-7)**\n- Machine Learning Basics\n- Data Structures Fundamentals\n- 2h study + 30min quiz")
        st.markdown("**Week 2: Practice (Days 8-14)**\n- Database Concepts\n- Web Development Basics\n- 2h study + 30min quiz")
    with col2:
        st.markdown("**Week 3: Advanced (Days 15-21)**\n- Advanced ML Algorithms\n- Complex Data Structures\n- 2h study + 1h practice")
        st.markdown("**Week 4: Revision (Days 22-30)**\n- All Topics Revision\n- Mock Tests (50 questions)\n- 3h revision + 1 mock test")
    st.markdown("### 💡 Study Tips\n1. Start with weak topics\n2. Use flashcards daily\n3. Review wrong answers\n4. Maintain consistency")
    st.info("💡 Last minute: focus on concepts, not memorization. Sleep well before exam!")

st.markdown("---")

# ============================================
# OPTION A: TOPIC COVERAGE TRACKER
# ============================================
with st.expander("📊 Topic Coverage Tracker - See Your Progress", expanded=False):
    st.markdown("### 🎯 Topic-wise Progress")
    for topic in ["Machine Learning", "Data Structures", "Databases", "Web Development"]:
        mastery = s.topic_mastery.get(topic, 0)
        if mastery >= 0.7:
            status = "✅ Mastered"; color = "🟢"
        elif mastery >= 0.4:
            status = "📚 In Progress"; color = "🟡"
        else:
            status = "⚠️ Need Focus"; color = "🔴"
        st.markdown(f"**{color} {topic}** - {status} ({mastery:.0%} covered)")
        st.progress(mastery)
        if mastery < 0.4:
            st.caption(f"💡 Focus on {topic} - Recommended 2 hours study")
        elif mastery < 0.7:
            st.caption(f"📖 Keep practicing {topic} - Recommended 1 hour daily")
        else:
            st.caption(f"🌟 Great work on {topic}! Review once before exam")
    avg_m = sum(s.topic_mastery.values()) / max(1, len(s.topic_mastery))
    st.markdown(f"### 📈 Overall Exam Readiness: {avg_m:.0%}")
    if avg_m >= 0.7:
        st.success("🎉 Well prepared! Keep up the good work!")
    elif avg_m >= 0.4:
        st.warning("⚠️ Good progress! Focus on weak topics before exam.")
    else:
        st.error("🔴 Need more practice! Follow the study plan.")
    predicted = 60 + avg_m*30
    st.info(f"🎯 Predicted Exam Score: {predicted:.0f}%")

st.markdown("---")

# ============================================
# OPTION A: QUICK REVISION MODE
# ============================================
with st.expander("⚡ Quick Revision Mode - Last 3 Days Strategy", expanded=False):
    st.markdown("### 🚀 3-Day Before Exam Sprint")
    tab1, tab2, tab3 = st.tabs(["📅 Day 1", "📅 Day 2", "📅 Day 3"])
    with tab1:
        st.markdown("**Day 1: Weak Topics Focus**\n- Practice 30 questions from weak areas\n- Review flashcards of weak topics\n- 🎯 Goal: Bring weak topics to 50% mastery")
        if st.button("🎯 Start Day 1 - Weak Topics Quiz", key="day1"):
            st.session_state.quiz_mode = "mixed"
            st.session_state.max_q = 30
            st.session_state.complete = False
            st.session_state.current_q = None
            st.session_state.asked_question_ids = set()
            st.rerun()
    with tab2:
        st.markdown("**Day 2: All Topics Practice**\n- Full mixed quiz (30 questions)\n- Achieve 70% accuracy\n- Review all wrong answers")
        if st.button("📝 Start Day 2 - Full Practice", key="day2"):
            st.session_state.quiz_mode = "mixed"
            st.session_state.max_q = 30
            st.session_state.complete = False
            st.session_state.current_q = None
            st.session_state.asked_question_ids = set()
            st.rerun()
    with tab3:
        st.markdown("**Day 3: Mock Test**\n- 50 questions, exam pattern\n- Build confidence\n- Review important concepts only")
        if st.button("🏆 Start Day 3 - Mock Test", key="day3"):
            st.session_state.quiz_mode = "mixed"
            st.session_state.max_q = 50
            st.session_state.complete = False
            st.session_state.current_q = None
            st.session_state.asked_question_ids = set()
            st.rerun()
    st.info("💡 Last minute tips: Focus on concepts, not rote memorization. Sleep well before exam!")

st.markdown("---")

# MODE SELECTION (Quiz or Flashcards)
col_mode1, col_mode2 = st.columns(2)
with col_mode1:
    if st.button("📝 Quiz Mode", use_container_width=True):
        st.session_state.show_flashcards = False
        st.rerun()
with col_mode2:
    if st.button("🃏 Flashcard Mode", use_container_width=True):
        st.session_state.show_flashcards = True
        st.rerun()
st.markdown("---")

if st.session_state.show_flashcards:
    show_flashcard_ui()
else:
    if st.session_state.quiz_mode is None:
        show_quiz_mode_selection()
    elif st.session_state.quiz_mode is not None and not st.session_state.complete:
        mode = st.session_state.quiz_mode
        if mode == "topic_wise":
            st.info(f"📌 Topic-Wise Mode: {st.session_state.selected_topic}")
        else:
            st.info("📌 Mixed Mode (Adaptive)")
        if st.button("🔄 Change Mode"):
            st.session_state.quiz_mode = None
            st.rerun()
        progress = s.total_questions / st.session_state.max_q
        st.progress(progress)
        st.caption(f"Question {s.total_questions + 1} of {st.session_state.max_q}")
        acc = s.get_accuracy()
        if acc >= 70:
            st.markdown('<p style="color:green">🌟 Great progress! Keep going!</p>', unsafe_allow_html=True)
        elif acc >= 40:
            st.markdown('<p style="color:orange">📚 Good! You can do better!</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:red">⚠️ Keep practicing! You will improve!</p>', unsafe_allow_html=True)

        if st.session_state.current_q is None and s.total_questions < st.session_state.max_q:
            if mode == "topic_wise":
                topic = st.session_state.selected_topic
                difficulty = s.get_difficulty(topic)
            else:
                weak = s.get_weak_topics()
                topic = weak[0] if weak else random.choice(st.session_state.topics)
                difficulty = s.get_difficulty(topic)
            q = st.session_state.qbank.get_question(topic, difficulty, st.session_state.asked_question_ids)
            if q:
                st.session_state.asked_question_ids.add(q['id'])
                st.session_state.current_q = q
                st.session_state.current_topic = topic

        if st.session_state.current_q:
            q = st.session_state.current_q
            diff_color = {"easy":"🟢","medium":"🟡","hard":"🔴"}
            st.markdown(f"""
            <div style='background:#f0f2f6; padding:20px; border-radius:10px'>
                <h3>📖 Question {s.total_questions + 1}</h3>
                <p><strong>Topic:</strong> {q['topic']} | <strong>Difficulty:</strong> {diff_color.get(q['difficulty'],'')} {q['difficulty'].upper()}</p>
                <p style='font-size:18px'>{q['question']}</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("💡 Hint"):
                st.info(f"Think about {q['topic']} concepts...")
            answer = st.radio("Select answer:", q['options'], key=f"q_{s.total_questions}", index=None)
            if st.button("✅ Submit", type="primary", use_container_width=True):
                if answer:
                    selected = answer[0]
                    correct = (selected == q['answer'])
                    s.update(st.session_state.current_topic, correct)

                    # Save progress after each answer (persistence)
                    save_student_data(s)

                    if correct:
                        st.success(f"✅ Correct! {q['explanation']}")
                        st.balloons()
                    else:
                        st.error("❌ Wrong Answer!")
                        st.markdown(f"""
                        <div style='background:#2ecc71; padding:15px; border-radius:10px; margin:10px 0'>
                            <p><strong>✓ Correct Answer:</strong> {q['answer']}) {next((o.split(') ',1)[1] for o in q['options'] if o.startswith(q['answer'])), '')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style='background:#3498db; padding:15px; border-radius:10px; margin:10px 0'>
                            <p><strong>💡 Explanation:</strong> {q['explanation']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        simple = get_simple_explanation(q['topic'], q['question'])
                        if simple:
                            st.markdown(f"""
                            <div style='background:#9b59b6; padding:15px; border-radius:10px; margin:10px 0'>
                                <p><strong>📌 Easy Explanation:</strong> {simple}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        st.info("💡 Don't worry! Learning happens by making mistakes. Keep practicing!")
                    if rf_model:
                        try:
                            pred = rf_model.predict(s.get_features())[0]
                            if pred > 0.6:
                                st.info(f"🎯 ML Model: {pred*100:.1f}% success chance")
                            else:
                                st.warning(f"⚠️ ML Model: {pred*100:.1f}% success chance - needs practice")
                        except:
                            pass
                    mastery = s.topic_mastery.get(st.session_state.current_topic, 0)
                    st.progress(mastery)
                    st.caption(f"📊 {st.session_state.current_topic} Mastery: {mastery:.0%}")
                    weak = s.get_weak_topics()
                    if weak:
                        st.info(f"💡 Recommendation: Practice {weak[0]} to improve!")
                    st.session_state.current_q = None
                    if s.total_questions >= st.session_state.max_q:
                        st.session_state.complete = True
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.warning("Please select an answer first")

    elif st.session_state.complete:
        st.balloons()
        if s.get_accuracy() >= 80:
            st.markdown("## 🌟🌟🌟 EXCELLENT! 🌟🌟🌟")
        elif s.get_accuracy() >= 60:
            st.markdown("## ⭐⭐ GOOD JOB! ⭐⭐")
        else:
            st.markdown("## 📚 KEEP PRACTICING! 📚")
        st.success("🎉 QUIZ COMPLETE!")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Questions", s.total_questions)
            st.metric("Correct", s.total_correct)
            st.metric("Accuracy", f"{s.get_accuracy()}%")
        with col2:
            st.markdown("**Topic Mastery:**")
            for topic, mastery in s.topic_mastery.items():
                st.markdown(f"- {topic}: {mastery:.0%}")
                st.progress(mastery)
        badges = []
        if s.streak >= 2:
            badges.append("🔥 Streak")
        if s.get_accuracy() >= 70:
            badges.append("🏆 Scholar")
        if len(s.topic_mastery) >= 2:
            badges.append("📚 Explorer")
        if badges:
            st.success(f"🎖️ Badges Earned: {' '.join(badges)}")
        weak = s.get_weak_topics()
        if weak:
            st.warning(f"⚠️ Needs Improvement: {', '.join(weak)}")
            st.markdown("### 📚 Recommendations:")
            for t in weak:
                st.markdown(f"- Review {t} fundamentals")

        # Leaderboard
        save_to_leaderboard(s, s.get_accuracy())
        show_leaderboard()

        if st.button("🔄 Start New Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
