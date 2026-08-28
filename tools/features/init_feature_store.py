import sqlite3
from pathlib import Path

def init_db():
    db_path = Path(__file__).parent.parent.parent / "data" / "processed" / "feature_store.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Table for Data Flywheel reports
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER,
        exercise TEXT,
        features_json TEXT,
        predicted_correct BOOLEAN,
        actual_correct BOOLEAN,
        user_feedback TEXT
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"Feature store initialized at {db_path}")

if __name__ == "__main__":
    init_db()
