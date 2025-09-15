from __future__ import annotations
import json
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sqlite3
import numpy as np
from .prompt_builder import CodeExemplar

@dataclass
class ExemplarEntry:
    """Database entry for code exemplars"""
    id: int
    task_hash: str
    code: str
    score: float
    metadata: Dict[str, Any]
    timestamp: float

class ExemplarDB:
    """
    Performance-indexed database with bucketed sampling for exemplar selection.
    Implements the paper's centered softmax bucketing strategy.
    """

    def __init__(self, db_path: str = "exemplars.db", num_buckets: int = 7):
        self.db_path = Path(db_path)
        self.num_buckets = num_buckets
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute('''
            CREATE TABLE IF NOT EXISTS exemplars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_hash TEXT NOT NULL,
                code TEXT NOT NULL,
                score REAL NOT NULL,
                metadata TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_task_hash ON exemplars(task_hash)')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_score ON exemplars(score)')
        conn.commit()
        conn.close()

    def add_exemplar(self, task_context: Dict[str, Any], code: str, score: float,
                    metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a new exemplar to the database"""
        import time
        task_hash = self._hash_task_context(task_context)
        metadata = metadata or {}

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            INSERT INTO exemplars (task_hash, code, score, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (task_hash, code, score, json.dumps(metadata), time.time()))
        exemplar_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return exemplar_id

    def get_exemplars_for_task(self, task_context: Dict[str, Any], n: int = 2) -> List[CodeExemplar]:
        """
        Get exemplars using bucketed sampling with centered softmax.

        As per paper: discretize scores into buckets, sample N distinct buckets
        via centered softmax over bucket means, then pick one code per bucket.
        """
        task_hash = self._hash_task_context(task_context)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute('''
            SELECT code, score, metadata FROM exemplars
            WHERE task_hash = ? ORDER BY score DESC
        ''', (task_hash,))

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return []

        # Extract scores and create buckets
        scores = [row[1] for row in rows]
        if len(scores) < n:
            # Not enough data for bucketing, return all
            return [CodeExemplar(row[0], row[1], json.loads(row[2])) for row in rows]

        # Discretize scores into buckets
        buckets = self._discretize_into_buckets(scores, rows)

        if len(buckets) < n:
            # Not enough buckets, return one from each
            return [self._sample_from_bucket(bucket) for bucket in buckets]

        # Sample N distinct buckets using centered softmax
        selected_buckets = self._sample_buckets_with_centered_softmax(buckets, n)

        # Pick one exemplar per selected bucket
        return [self._sample_from_bucket(bucket) for bucket in selected_buckets]

    def _hash_task_context(self, task_context: Dict[str, Any]) -> str:
        """Create a hash for task context to group similar tasks"""
        # Sort keys for consistent hashing
        canonical = json.dumps(task_context, sort_keys=True)
        import hashlib
        return hashlib.md5(canonical.encode()).hexdigest()

    def _discretize_into_buckets(self, scores: List[float], rows: List[Tuple]) -> List[List[Tuple]]:
        """Discretize scores into buckets"""
        if not scores:
            return []

        min_score, max_score = min(scores), max(scores)
        if min_score == max_score:
            return [rows]  # All same score, single bucket

        # Create bucket boundaries
        bucket_width = (max_score - min_score) / self.num_buckets
        buckets = [[] for _ in range(self.num_buckets)]

        for i, (code, score, metadata) in enumerate(rows):
            bucket_idx = min(int((score - min_score) / bucket_width), self.num_buckets - 1)
            buckets[bucket_idx].append((code, score, metadata))

        # Filter out empty buckets
        return [bucket for bucket in buckets if bucket]

    def _sample_buckets_with_centered_softmax(self, buckets: List[List[Tuple]], n: int) -> List[List[Tuple]]:
        """
        Sample N distinct buckets using centered softmax over bucket means.
        Implements the paper's strategy for competitive + diverse sampling.
        """
        if len(buckets) <= n:
            return buckets

        # Compute bucket means
        bucket_means = []
        for bucket in buckets:
            scores = [row[1] for row in bucket]
            bucket_means.append(sum(scores) / len(scores))

        # Center the means (subtract mean of means)
        overall_mean = sum(bucket_means) / len(bucket_means)
        centered_means = [m - overall_mean for m in bucket_means]

        # Apply softmax to centered means
        exp_means = [math.exp(m) for m in centered_means]
        softmax_sum = sum(exp_means)
        probs = [e / softmax_sum for e in exp_means]

        # Sample N distinct buckets without replacement
        selected_indices = np.random.choice(
            len(buckets), size=n, replace=False, p=probs
        ).tolist()

        return [buckets[i] for i in selected_indices]

    def _sample_from_bucket(self, bucket: List[Tuple]) -> CodeExemplar:
        """Sample one exemplar from a bucket (randomly for diversity)"""
        code, score, metadata = random.choice(bucket)
        return CodeExemplar(code, score, json.loads(metadata))

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)

        # Total count
        total = conn.execute('SELECT COUNT(*) FROM exemplars').fetchone()[0]

        # Score statistics
        score_stats = conn.execute('''
            SELECT MIN(score), MAX(score), AVG(score) FROM exemplars
        ''').fetchone()

        # Task diversity
        unique_tasks = conn.execute('SELECT COUNT(DISTINCT task_hash) FROM exemplars').fetchone()[0]

        conn.close()

        return {
            'total_exemplars': total,
            'unique_tasks': unique_tasks,
            'min_score': score_stats[0],
            'max_score': score_stats[1],
            'avg_score': score_stats[2]
        }

    def cleanup_old_exemplars(self, max_age_days: int = 30, max_per_task: int = 100):
        """Clean up old or excessive exemplars"""
        import time
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        conn = sqlite3.connect(self.db_path)

        # Remove old exemplars
        conn.execute('DELETE FROM exemplars WHERE timestamp < ?', (cutoff_time,))

        # Limit exemplars per task (keep top performers)
        conn.execute('''
            DELETE FROM exemplars WHERE id NOT IN (
                SELECT id FROM (
                    SELECT id, ROW_NUMBER() OVER (
                        PARTITION BY task_hash ORDER BY score DESC
                    ) as rn FROM exemplars
                ) WHERE rn <= ?
            )
        ''', (max_per_task,))

        conn.commit()
        conn.close()