"""
ADAPTIVE LEARNING SYSTEM - 50 QUESTIONS + FLASHCARDS + STUDY PLAN
M.Tech Project

Topics: Machine Learning, Data Structures, Databases, Web Development
Total Questions: 50
Total Flashcards: 16 (4 per topic)
"""

import streamlit as st
import pandas as pd
import numpy as np
import random
import joblib

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Adaptive Learning System",
    page_icon="🎓",
    layout="wide"
)

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('adaptive_learning_model.pkl')
        return model
    except:
        return None

rf_model = load_model()

# ============================================
# SIMPLE EXPLANATIONS FOR WRONG ANSWERS
# ============================================

def get_simple_explanation(topic, question):
    """Return simple language explanation for wrong answers"""
    
    explanations = {
        "Machine Learning": {
            "What does supervised learning mean?": "Jaise teacher answer sheet dekar padhata hai. Model ko pata hai ki output kya hona chahiye.",
            "What is unsupervised learning?": "Jaise bina answer sheet ke khud patterns find karna. Model ko kuch nahi pata.",
            "What is overfitting?": "Jaise exam ke questions rat lo lekin naye questions solve na kar pao.",
            "Which algorithm minimizes sum of squared residuals?": "Linear Regression line draw karta hai jo sab points ke beech se guzare. Isliye yeh best fit line banata hai.",
            "What is classification?": "Yes/No ya Category batana. Jaise 'Ye email spam hai ya nahi?'"
        },
        "Data Structures": {
            "Which data structure follows LIFO?": "Stack. Jaise plate ka stack - upar rakho, upar se uthao.",
            "Which data structure follows FIFO?": "Queue. Jaise ticket line - pehle aao, pehle jao.",
            "Time complexity of binary search?": "O(log n). Jaise dictionary mein word dhundhna - aadha karo, phir aadha.",
            "What data structure is used for recursion?": "Stack. Jaise ek function doosre function ko call kare, sab wait karte hain."
        },
        "Databases": {
            "What does SQL stand for?": "Structured Query Language. Database se baat karne ki bhasha.",
            "What is a primary key?": "Unique number. Jaise Aadhar Card - har ek unique hota hai.",
            "What is a foreign key?": "Do tables ko jodne wala setu. Jaise order table mein customer id.",
            "Which normal form removes transitive dependency?": "3NF. Data repeat hone se bachne ka tarika."
        },
        "Web Development": {
            "What does HTML stand for?": "Hyper Text Markup Language. Website ka structure banata hai.",
            "What does CSS stand for?": "Cascading Style Sheets. Website ko sundar banata hai.",
            "What is JavaScript?": "Website mein interactivity laane wali language. Button click, animation.",
            "Frontend vs Backend difference?": "Frontend = Dining area (jo customer dekhta hai). Backend = Kitchen (data process hota hai)."
        }
    }
    
    topic_dict = explanations.get(topic, {})
    return topic_dict.get(question, None)


# ============================================
# STUDENT MODEL
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
        else:
            new = max(0.0, current - 0.1)
        self.topic_mastery[topic] = round(new, 2)
        
        self.total_questions += 1
        if correct:
            self.total_correct += 1
            self.streak += 1
        else:
            self.streak = 0
    
    def get_difficulty(self, topic):
        mastery = self.topic_mastery.get(topic, 0.3)
        if mastery >= 0.7:
            return "hard"
        elif mastery >= 0.4:
            return "medium"
        return "easy"
    
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
# QUESTION BANK (50 QUESTIONS)
# ============================================
class QuestionBank:
    def __init__(self):
        self.questions = self._load()
    
    def _load(self):
        return [
            # ========== MACHINE LEARNING (13 questions) ==========
            {"id": 1, "topic": "Machine Learning", "difficulty": "easy",
             "question": "What does supervised learning mean?",
             "options": ["A) Learning with labeled data", "B) Learning without data", "C) Learning with images", "D) Learning with rewards"],
             "answer": "A", "explanation": "Supervised learning uses labeled input-output pairs."},
            
            {"id": 2, "topic": "Machine Learning", "difficulty": "easy",
             "question": "What is unsupervised learning?",
             "options": ["A) Find patterns in unlabeled data", "B) Predict labels", "C) Classify images", "D) Play games"],
             "answer": "A", "explanation": "Unsupervised learning finds hidden patterns without labels."},
            
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
             "options": ["A) Predicting categories", "B) Predicting numbers", "C) Clustering", "D) Reducing dimensions"],
             "answer": "A", "explanation": "Classification predicts discrete categories."},
            
            {"id": 9, "topic": "Machine Learning", "difficulty": "medium",
             "question": "What is regression?",
             "options": ["A) Predicting continuous values", "B) Predicting categories", "C) Grouping data", "D) Finding outliers"],
             "answer": "A", "explanation": "Regression predicts continuous values like price."},
            
            {"id": 10, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is the kernel trick in SVM?",
             "options": ["A) Map data to higher dimension", "B) Reduce dimensions", "C) Remove outliers", "D) Speed up training"],
             "answer": "A", "explanation": "Kernel trick maps to higher dimension without explicit computation."},
            
            {"id": 11, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is the bias-variance tradeoff?",
             "options": ["A) Low bias = high variance", "B) Low bias = low variance", "C) High bias = high variance", "D) No relationship"],
             "answer": "A", "explanation": "Simple models = high bias, complex = high variance."},
            
            {"id": 12, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is ensemble learning?",
             "options": ["A) Combining multiple models", "B) Single model", "C) Deep learning", "D) Reinforcement learning"],
             "answer": "A", "explanation": "Ensemble combines multiple models (Random Forest)."},
            
            {"id": 13, "topic": "Machine Learning", "difficulty": "hard",
             "question": "What is cross-validation?",
             "options": ["A) Splitting data multiple times", "B) Single split", "C) Preprocessing", "D) Feature engineering"],
             "answer": "A", "explanation": "Cross-validation ensures model generalizes well."},
            
            # ========== DATA STRUCTURES (12 questions) ==========
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
             "options": ["A) Collection of same type", "B) Different types", "C) Key-value pairs", "D) Tree structure"],
             "answer": "A", "explanation": "Array stores elements of same data type."},
            
            {"id": 17, "topic": "Data Structures", "difficulty": "easy",
             "question": "What is a linked list?",
             "options": ["A) Nodes connected by pointers", "B) Continuous memory", "C) Key-value pairs", "D) Tree"],
             "answer": "A", "explanation": "Linked list has nodes with data and next pointer."},
            
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
             "options": ["A) Max 2 children per node", "B) 3 children", "C) Linear", "D) No children"],
             "answer": "A", "explanation": "Binary tree nodes have at most 2 children."},
            
            {"id": 21, "topic": "Data Structures", "difficulty": "medium",
             "question": "What is a hash table?",
             "options": ["A) Key-value with hash function", "B) Array", "C) Linked", "D) Tree"],
             "answer": "A", "explanation": "Hash table uses hash function for O(1) access."},
            
            {"id": 22, "topic": "Data Structures", "difficulty": "hard",
             "question": "What keeps Red-Black Tree balanced?",
             "options": ["A) All red", "B) Color properties", "C) Complete", "D) Sorted"],
             "answer": "B", "explanation": "Red-Black Tree uses coloring rules."},
            
            {"id": 23, "topic": "Data Structures", "difficulty": "hard",
             "question": "Worst-case time of QuickSort?",
             "options": ["A) O(n log n)", "B) O(n²)", "C) O(n)", "D) O(log n)"],
             "answer": "B", "explanation": "QuickSort worst case O(n²)."},
            
            {"id": 24, "topic": "Data Structures", "difficulty": "hard",
             "question": "What is a B-tree?",
             "options": ["A) Self-balancing for databases", "B) Binary tree", "C) Hash table", "D) Linked list"],
             "answer": "A", "explanation": "B-tree used in databases and file systems."},
            
            {"id": 25, "topic": "Data Structures", "difficulty": "hard",
             "question": "Dijkstra's algorithm used for?",
             "options": ["A) Shortest path", "B) Sorting", "C) Searching", "D) Hashing"],
             "answer": "A", "explanation": "Dijkstra finds shortest path in graphs."},
            
            # ========== DATABASES (12 questions) ==========
            {"id": 26, "topic": "Databases", "difficulty": "easy",
             "question": "What does SQL stand for?",
             "options": ["A) Structured Query Language", "B) Simple Query Logic", "C) System Query Language", "D) Sorted Query Language"],
             "answer": "A", "explanation": "SQL = Structured Query Language."},
            
            {"id": 27, "topic": "Databases", "difficulty": "easy",
             "question": "What is a primary key?",
             "options": ["A) Unique record identifier", "B) Duplicate values", "C) Foreign reference", "D) Index only"],
             "answer": "A", "explanation": "Primary key uniquely identifies each record."},
            
            {"id": 28, "topic": "Databases", "difficulty": "easy",
             "question": "What is a foreign key?",
             "options": ["A) Reference to another table's key", "B) Primary key", "C) Unique key", "D) Composite key"],
             "answer": "A", "explanation": "Foreign key links two tables together."},
            
            {"id": 29, "topic": "Databases", "difficulty": "easy",
             "question": "What is a table in database?",
             "options": ["A) Rows and columns", "B) Single row", "C) Single column", "D) Database"],
             "answer": "A", "explanation": "Table stores data in rows and columns."},
            
            {"id": 30, "topic": "Databases", "difficulty": "medium",
             "question": "Which normal form removes transitive dependency?",
             "options": ["A) 1NF", "B) 2NF", "C) 3NF", "D) BCNF"],
             "answer": "C", "explanation": "3NF removes transitive dependencies."},
            
            {"id": 31, "topic": "Databases", "difficulty": "medium",
             "question": "What is a JOIN?",
             "options": ["A) Combine tables", "B) Delete records", "C) Create tables", "D) Update records"],
             "answer": "A", "explanation": "JOIN combines data from multiple tables."},
            
            {"id": 32, "topic": "Databases", "difficulty": "medium",
             "question": "What is an index?",
             "options": ["A) Faster data retrieval", "B) Data storage", "C) Backup", "D) Encryption"],
             "answer": "A", "explanation": "Index speeds up data retrieval."},
            
            {"id": 33, "topic": "Databases", "difficulty": "medium",
             "question": "What is a query?",
             "options": ["A) Request for data", "B) Deletion", "C) Creation", "D) Backup"],
             "answer": "A", "explanation": "Query retrieves specific data from database."},
            
            {"id": 34, "topic": "Databases", "difficulty": "hard",
             "question": "What does ACID stand for?",
             "options": ["A) Atomicity,Consistency,Isolation,Durability", "B) Atomicity,Concurrency,Integrity,Durability", "C) Availability,Consistency,Isolation,Durability", "D) Atomicity,Consistency,Integrity,Durability"],
             "answer": "A", "explanation": "ACID ensures reliable database transactions."},
            
            {"id": 35, "topic": "Databases", "difficulty": "hard",
             "question": "INNER JOIN vs LEFT JOIN?",
             "options": ["A) INNER = matching only, LEFT = all from left", "B) Both same", "C) LEFT = matching only", "D) INNER = all rows"],
             "answer": "A", "explanation": "INNER JOIN returns matching rows; LEFT JOIN all from left."},
            
            {"id": 36, "topic": "Databases", "difficulty": "hard",
             "question": "What is a transaction?",
             "options": ["A) Unit of work", "B) Table creation", "C) Index creation", "D) Backup"],
             "answer": "A", "explanation": "Transaction is a unit of work with ACID."},
            
            {"id": 37, "topic": "Databases", "difficulty": "hard",
             "question": "What is NoSQL?",
             "options": ["A) Non-relational database", "B) SQL only", "C) Relational", "D) Programming"],
             "answer": "A", "explanation": "NoSQL databases are non-relational (MongoDB)."},
            
            # ========== WEB DEVELOPMENT (13 questions) ==========
            {"id": 38, "topic": "Web Development", "difficulty": "easy",
             "question": "What does HTML stand for?",
             "options": ["A) Hyper Text Markup Language", "B) High Tech Modern Language", "C) Hyper Transfer Markup Language", "D) Home Tool Markup Language"],
             "answer": "A", "explanation": "HTML = Hyper Text Markup Language."},
            
            {"id": 39, "topic": "Web Development", "difficulty": "easy",
             "question": "What does CSS stand for?",
             "options": ["A) Cascading Style Sheets", "B) Computer Style Sheets", "C) Creative Style Sheets", "D) Colorful Style Sheets"],
             "answer": "A", "explanation": "CSS controls web page appearance."},
            
            {"id": 40, "topic": "Web Development", "difficulty": "easy",
             "question": "What is JavaScript?",
             "options": ["A) Programming language for web", "B) CSS framework", "C) Database", "D) HTML tag"],
             "answer": "A", "explanation": "JavaScript adds interactivity to web pages."},
            
            {"id": 41, "topic": "Web Development", "difficulty": "easy",
             "question": "What is a web browser?",
             "options": ["A) Software to view websites", "B) Web server", "C) Database", "D) Language"],
             "answer": "A", "explanation": "Browser (Chrome, Firefox) displays web pages."},
            
            {"id": 42, "topic": "Web Development", "difficulty": "medium",
             "question": "Frontend vs Backend difference?",
             "options": ["A) Frontend = client, Backend = server", "B) Both same", "C) Frontend = database", "D) Frontend = server"],
             "answer": "A", "explanation": "Frontend is what users see; Backend handles data."},
            
            {"id": 43, "topic": "Web Development", "difficulty": "medium",
             "question": "What is a framework?",
             "options": ["A) Reusable code structure", "B) Language", "C) Database", "D) Server"],
             "answer": "A", "explanation": "Frameworks provide reusable code templates."},
            
            {"id": 44, "topic": "Web Development", "difficulty": "medium",
             "question": "What is an API?",
             "options": ["A) Application Programming Interface", "B) Application Protocol Interface", "C) Advanced Programming Interface", "D) Automated Program Interface"],
             "answer": "A", "explanation": "API allows different software to communicate."},
            
            {"id": 45, "topic": "Web Development", "difficulty": "medium",
             "question": "What is HTTP?",
             "options": ["A) Hypertext Transfer Protocol", "B) High Transfer Protocol", "C) Hyper Text Program", "D) High Text Protocol"],
             "answer": "A", "explanation": "HTTP is protocol for web communication."},
            
            {"id": 46, "topic": "Web Development", "difficulty": "medium",
             "question": "What is a responsive website?",
             "options": ["A) Works on all devices", "B) Desktop only", "C) Mobile only", "D) Tablet only"],
             "answer": "A", "explanation": "Responsive design adapts to different screen sizes."},
            
            {"id": 47, "topic": "Web Development", "difficulty": "hard",
             "question": "GET vs POST difference?",
             "options": ["A) GET = URL, POST = body", "B) Both same", "C) POST = URL", "D) GET = body"],
             "answer": "A", "explanation": "GET exposes data in URL; POST hides in body."},
            
            {"id": 48, "topic": "Web Development", "difficulty": "hard",
             "question": "What is a REST API?",
             "options": ["A) Architectural style for web services", "B) Language", "C) Database", "D) Server"],
             "answer": "A", "explanation": "REST API follows stateless client-server architecture."},
            
            {"id": 49, "topic": "Web Development", "difficulty": "hard",
             "question": "What is the DOM?",
             "options": ["A) Document Object Model", "B) Data Object Model", "C) Document Oriented Model", "D) Data Oriented Model"],
             "answer": "A", "explanation": "DOM represents web page structure as objects."},
            
            {"id": 50, "topic": "Web Development", "difficulty": "hard",
             "question": "What is a Single Page Application (SPA)?",
             "options": ["A) Single HTML, content dynamically updated", "B) Multiple HTML pages", "C) Server-side rendering", "D) Static website"],
             "answer": "A", "explanation": "SPA loads one HTML and updates content dynamically."},
        ]
    
    def get_question(self, topic, difficulty):
        pool = [q for q in self.questions if q["topic"] == topic and q["difficulty"] == difficulty]
        if not pool:
            pool = [q for q in self.questions if q["topic"] == topic]
        return random.choice(pool) if pool else None
    
    def get_topics(self):
        return list(set(q["topic"] for q in self.questions))


# ============================================
# FLASHCARD DATA (16 Cards - 4 per topic)
# ============================================

flashcards_data = {
    "Machine Learning": [
        {"question": "What is Supervised Learning?", 
         "answer": "Learning with labeled input-output pairs.",
         "simple_explanation": "📌 Jaise teacher answer sheet dekar padhata hai."},
        {"question": "What is Unsupervised Learning?", 
         "answer": "Learning without labeled data.",
         "simple_explanation": "📌 Jaise bina answer sheet ke khud patterns find karna."},
        {"question": "What is Overfitting?", 
         "answer": "Model learns training data too well, fails on new data.",
         "simple_explanation": "📌 Jaise exam ke questions rat lo lekin naye questions solve na kar pao."},
        {"question": "Classification vs Regression?", 
         "answer": "Classification = categories, Regression = continuous values.",
         "simple_explanation": "📌 Classification = Yes/No, Regression = Price."}
    ],
    "Data Structures": [
        {"question": "What is a Stack?", 
         "answer": "LIFO data structure.",
         "simple_explanation": "📌 Jaise plate ka stack - upar rakho, upar se uthao."},
        {"question": "What is a Queue?", 
         "answer": "FIFO data structure.",
         "simple_explanation": "📌 Jaise ticket line - pehle aao, pehle jao."},
        {"question": "What is Binary Search?", 
         "answer": "O(log n) algorithm.",
         "simple_explanation": "📌 Jaise dictionary mein word dhundhna - aadha karo, phir aadha."},
        {"question": "What is a Linked List?", 
         "answer": "Nodes connected by pointers.",
         "simple_explanation": "📌 Jaise train ke coaches - ek coach doosre se connected."}
    ],
    "Databases": [
        {"question": "What is SQL?", 
         "answer": "Structured Query Language.",
         "simple_explanation": "📌 Jaise database se baat karne ki bhasha."},
        {"question": "What is a Primary Key?", 
         "answer": "Unique record identifier.",
         "simple_explanation": "📌 Jaise Aadhar Card - har ek unique."},
        {"question": "What is a Foreign Key?", 
         "answer": "Links to primary key of another table.",
         "simple_explanation": "📌 Jaise 2 tables ko jodne wala setu."},
        {"question": "What is Normalization?", 
         "answer": "Organizing data to reduce redundancy.",
         "simple_explanation": "📌 Jaise data repeat hone se bachne ka tarika."}
    ],
    "Web Development": [
        {"question": "What is HTML?", 
         "answer": "Structure of web pages.",
         "simple_explanation": "📌 Jaise ghar ka naksha - structure banata hai."},
        {"question": "What is CSS?", 
         "answer": "Styling of web pages.",
         "simple_explanation": "📌 Jaise ghar ki painting - sundar banata hai."},
        {"question": "What is JavaScript?", 
         "answer": "Interactivity for web pages.",
         "simple_explanation": "📌 Jaise ghar mein lights on/off - functionality deta hai."},
        {"question": "Frontend vs Backend?", 
         "answer": "Frontend = browser, Backend = server.",
         "simple_explanation": "📌 Frontend = dining area, Backend = kitchen."}
    ]
}


# ============================================
# FLASHCARD MANAGER CLASS
# ============================================

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


# ============================================
# FLASHCARD UI
# ============================================

def show_flashcard_ui():
    """Display flashcard interface"""
    
    if 'flashcard_manager' not in st.session_state:
        st.session_state.flashcard_manager = FlashcardManager()
    
    mgr = st.session_state.flashcard_manager
    
    if mgr.current_topic is None:
        st.subheader("📚 Choose a Topic for Flashcards")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🤖 Machine Learning")
            st.caption("AI, algorithms, predictions")
            if st.button("Start ML Flashcards", key="ml_fc", use_container_width=True):
                mgr.set_topic("Machine Learning")
                st.rerun()
            
            st.markdown("### 🗄️ Databases")
            st.caption("SQL, tables, queries")
            if st.button("Start DB Flashcards", key="db_fc", use_container_width=True):
                mgr.set_topic("Databases")
                st.rerun()
        
        with col2:
            st.markdown("### 📊 Data Structures")
            st.caption("Stack, Queue, Trees")
            if st.button("Start DS Flashcards", key="ds_fc", use_container_width=True):
                mgr.set_topic("Data Structures")
                st.rerun()
            
            st.markdown("### 🌐 Web Development")
            st.caption("HTML, CSS, JavaScript")
            if st.button("Start Web Flashcards", key="web_fc", use_container_width=True):
                mgr.set_topic("Web Development")
                st.rerun()
        
        if st.button("🔙 Back", use_container_width=True):
            st.session_state.show_flashcards = False
            st.rerun()
    
    else:
        st.markdown(f"""
        <div style='background:#3498db; padding:15px; border-radius:10px'>
            <h2 style='color:white'>📚 {mgr.current_topic}</h2>
            <p style='color:white'>Card {mgr.get_progress()}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.progress(mgr.current_card_index / max(1, mgr.total_cards()))
        
        card = mgr.get_current_card()
        if card:
            st.markdown(f"""
            <div style='background:#f0f2f6; padding:40px; border-radius:15px; text-align:center'>
                <h3>❓ {card['question']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if not mgr.revealed:
                if st.button("🔓 Reveal Answer", type="primary", use_container_width=True):
                    mgr.reveal()
                    st.rerun()
            
            if mgr.revealed:
                st.markdown(f"""
                <div style='background:#2ecc71; padding:15px; border-radius:10px'>
                    <p><strong>✓ Answer:</strong> {card['answer']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style='background:#3498db; padding:15px; border-radius:10px'>
                    <p><strong>💡 Easy Explanation:</strong> {card['simple_explanation']}</p>
                </div>
                """, unsafe_allow_html=True)
                
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
                        if not mgr.next_card():
                            st.success("🎉 Completed all flashcards!")
                            st.balloons()
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
            st.rerun()
    
    with col2:
        st.markdown("### 🔄 MIXED QUIZ (Adaptive)\nQuestions from all topics.\nSystem will mix questions and adapt difficulty.")
        if st.button("Start Mixed Quiz", type="primary"):
            st.session_state.quiz_mode = "mixed"
            st.session_state.selected_topic = None
            st.session_state.complete = False
            st.session_state.current_q = None
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
    st.session_state.qbank = None
if 'topics' not in st.session_state:
    st.session_state.topics = []
if 'quiz_mode' not in st.session_state:
    st.session_state.quiz_mode = None
if 'show_flashcards' not in st.session_state:
    st.session_state.show_flashcards = False


# ============================================
# PAGE 1: ONLY REGISTRATION
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
            st.session_state.student = SimpleStudent(sid, name)
            st.session_state.qbank = QuestionBank()
            st.session_state.topics = st.session_state.qbank.get_topics()
            st.session_state.complete = False
            st.session_state.current_q = None
            st.session_state.quiz_mode = None
            st.rerun()
        else:
            st.error("Please enter both ID and Name")
    
    st.markdown("---")
    st.markdown("© Adaptive Learning System | M.Tech Project")


# ============================================
# PAGE 2: AFTER LOGIN
# ============================================
elif st.session_state.student:
    
    s = st.session_state.student
    
    st.title("🎓 AI-Based Adaptive Learning System")
    st.markdown("---")
    
    st.success(f"✅ Welcome {s.name} (ID: {s.student_id})")
    
    # ============================================
    # RESEARCH DASHBOARD
    # ============================================
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
        st.dataframe(pd.DataFrame(imp_data), use_container_width=True)
        
        chart_df = pd.DataFrame({"Feature": imp_data["Feature"], "Importance (%)": imp_data["Importance (%)"]})
        st.bar_chart(chart_df.set_index("Feature"))
        
        for feat, imp in zip(imp_data["Feature"], imp_data["Importance (%)"]):
            st.markdown(f"**{feat}** - {imp}%")
            st.progress(imp/100)
        
        st.success("💡 Behavioral features (91.3%) are MORE predictive than quiz scores (8.7%)!")
    
    st.markdown("---")
    
    # ============================================
    # OPTION A: EXAM PREPARATION GUIDE
    # ============================================
    with st.expander("📅 Exam Preparation Guide - How to Cover All Topics", expanded=False):
        st.markdown("### 🎯 30-Day Study Plan for Engineering Exam")
        
        # Safe way to get student mastery data (even if no questions answered)
        if s and hasattr(s, 'topic_mastery'):
            topic_mastery = s.topic_mastery
            mastered_topics = len([t for t, m in topic_mastery.items() if m >= 0.7])
            weak_topics = [t for t, m in topic_mastery.items() if m < 0.4]
        else:
            topic_mastery = {}
            mastered_topics = 0
            weak_topics = []
        
        total_topics = len(st.session_state.topics) if st.session_state.topics else 4
        
        st.markdown("**📊 Your Current Progress:**")
        st.write(f"✅ Topics Mastered: {mastered_topics}/{total_topics}")
        st.write(f"⚠️ Topics Needing Attention: {len(weak_topics)}/{total_topics}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Week 1: Foundation (Days 1-7)**")
            st.markdown("- 📚 Machine Learning Basics")
            st.markdown("- 📊 Data Structures Fundamentals")
            st.markdown("- ✅ Daily: 2 hours study + 30 min quiz")
            st.markdown("")
            st.markdown("**Week 2: Practice (Days 8-14)**")
            st.markdown("- 🗄️ Database Concepts")
            st.markdown("- 🌐 Web Development Basics")
            st.markdown("- ✅ Daily: 2 hours study + 30 min quiz")
        
        with col2:
            st.markdown("**Week 3: Advanced (Days 15-21)**")
            st.markdown("- 🤖 Advanced ML Algorithms")
            st.markdown("- 📈 Complex Data Structures")
            st.markdown("- ✅ Daily: 2 hours study + 1 hour practice")
            st.markdown("")
            st.markdown("**Week 4: Revision (Days 22-30)**")
            st.markdown("- 🔄 All Topics Revision")
            st.markdown("- 📝 Mock Tests (50 questions)")
            st.markdown("- ✅ Daily: 3 hours revision + 1 mock test")
        
        st.markdown("---")
        st.markdown("### 💡 Study Tips")
        st.markdown("1. **Start with weak topics** (prioritize)")
        st.markdown("2. **Use flashcards daily** (20 min)")
        st.markdown("3. **Take weekly quizzes** (track progress)")
        st.markdown("4. **Review wrong answers** (learn from mistakes)")
        st.markdown("5. **Maintain consistency** (daily study)")
        
        st.info("💡 **Last Minute Tips:** Focus on concepts, not memorization. Sleep well before exam!")
    
    st.markdown("---")
    
    # ============================================
    # OPTION A: TOPIC COVERAGE TRACKER
    # ============================================
    with st.expander("📊 Topic Coverage Tracker - See Your Progress", expanded=False):
        st.markdown("### 🎯 Topic-wise Progress")
        
        topics = ["Machine Learning", "Data Structures", "Databases", "Web Development"]
        
        for topic in topics:
            mastery = s.topic_mastery.get(topic, 0)
            
            if mastery >= 0.7:
                status = "✅ Mastered"
                color = "🟢"
            elif mastery >= 0.4:
                status = "📚 In Progress"
                color = "🟡"
            else:
                status = "⚠️ Need Focus"
                color = "🔴"
            
            st.markdown(f"**{color} {topic}** - {status} ({mastery:.0%} covered)")
            st.progress(mastery)
            
            if mastery < 0.4:
                st.caption(f"💡 Focus on {topic} - Recommended 2 hours study")
            elif mastery < 0.7:
                st.caption(f"📖 Keep practicing {topic} - Recommended 1 hour daily")
            else:
                st.caption(f"🌟 Great work on {topic}! Review once before exam")
        
        avg_mastery = sum(s.topic_mastery.values()) / max(1, len(s.topic_mastery))
        
        st.markdown("---")
        st.markdown(f"### 📈 Overall Exam Readiness: {avg_mastery:.0%}")
        
        if avg_mastery >= 0.7:
            st.success("🎉 You are well prepared! Keep up the good work!")
        elif avg_mastery >= 0.4:
            st.warning("⚠️ Good progress! Focus on weak topics before exam.")
        else:
            st.error("🔴 Need more practice! Follow the study plan above.")
        
        predicted_score = 60 + (avg_mastery * 30)
        st.info(f"🎯 Predicted Exam Score: {predicted_score:.0f}% (based on your current mastery)")
    
    st.markdown("---")
    
    # ============================================
    # OPTION A: QUICK REVISION MODE
    # ============================================
    with st.expander("⚡ Quick Revision Mode - Last 3 Days Strategy", expanded=False):
        st.markdown("### 🚀 3-Day Before Exam Sprint")
        
        tab1, tab2, tab3 = st.tabs(["📅 Day 1", "📅 Day 2", "📅 Day 3"])
        
        with tab1:
            st.markdown("""
            **Day 1: Weak Topics Focus**
            - ✅ Review all weak topics (identified by ML model)
            - ✅ Practice 30 questions from weak areas
            - ✅ Review flashcards of weak topics
            - 🎯 Goal: Bring weak topics to 50% mastery
            """)
            if st.button("🎯 Start Day 1 - Weak Topics Quiz", key="day1"):
                st.session_state.quiz_mode = "mixed"
                st.session_state.max_q = 30
                st.session_state.complete = False
                st.session_state.current_q = None
                st.rerun()
        
        with tab2:
            st.markdown("""
            **Day 2: All Topics Practice**
            - ✅ Revise strong topics (30 min)
            - ✅ Practice medium difficulty questions
            - ✅ Take full mixed quiz (30 questions)
            - 🎯 Goal: Achieve 70% accuracy overall
            """)
            if st.button("📝 Start Day 2 - Full Practice", key="day2"):
                st.session_state.quiz_mode = "mixed"
                st.session_state.max_q = 30
                st.session_state.complete = False
                st.session_state.current_q = None
                st.rerun()
        
        with tab3:
            st.markdown("""
            **Day 3: Final Revision**
            - ✅ Quick flashcards review (all topics)
            - ✅ Mock test similar to exam pattern
            - ✅ Review important concepts only
            - 🎯 Goal: Build confidence before exam
            """)
            if st.button("🏆 Start Day 3 - Mock Test", key="day3"):
                st.session_state.quiz_mode = "mixed"
                st.session_state.max_q = 50
                st.session_state.complete = False
                st.session_state.current_q = None
                st.rerun()
        
        st.info("💡 **Last Minute Tips:** Focus on concepts, not memorization. Sleep well before exam!")
    
    st.markdown("---")
    
    # ============================================
    # MODE SELECTION (Quiz or Flashcards)
    # ============================================
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
    
    # ============================================
    # FLASHCARD MODE
    # ============================================
    if st.session_state.show_flashcards:
        show_flashcard_ui()
    
    # ============================================
    # QUIZ MODE
    # ============================================
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
                
                q = st.session_state.qbank.get_question(topic, difficulty)
                st.session_state.current_q = q
                st.session_state.current_topic = topic
            
            if st.session_state.current_q:
                q = st.session_state.current_q
                diff_color = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}
                
                st.markdown(f"""
                <div style='background:#f0f2f6; padding:20px; border-radius:10px'>
                    <h3>📖 Question {s.total_questions + 1}</h3>
                    <p><strong>Topic:</strong> {q['topic']} | <strong>Difficulty:</strong> {diff_color.get(q['difficulty'], "")} {q['difficulty'].upper()}</p>
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
                        
                        if correct:
                            st.success(f"✅ Correct! {q['explanation']}")
                            st.balloons()
                        else:
                            st.error(f"❌ Wrong! Correct answer is '{q['answer']}'")
                            
                            st.markdown(f"""
                            <div style='background:#2ecc71; padding:15px; border-radius:10px; margin:10px 0'>
                                <p><strong>✓ Correct Answer:</strong> {q['answer']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"""
                            <div style='background:#3498db; padding:15px; border-radius:10px; margin:10px 0'>
                                <p><strong>💡 Explanation:</strong> {q['explanation']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            simple_exp = get_simple_explanation(q['topic'], q['question'])
                            if simple_exp:
                                st.markdown(f"""
                                <div style='background:#9b59b6; padding:15px; border-radius:10px; margin:10px 0'>
                                    <p><strong>📌 Easy Explanation:</strong> {simple_exp}</p>
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
                        st.rerun()
                    else:
                        st.warning("Please select an answer first")
        
        elif st.session_state.complete:
            
            if s.get_accuracy() >= 80:
                st.markdown("## 🌟🌟🌟 EXCELLENT! 🌟🌟🌟")
                st.balloons()
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
            
            if st.button("🔄 Start New Session", use_container_width=True):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()