"""
LegalComparator — Indonesian Anti-Corruption Law NLP Model
===========================================================
Model untuk membandingkan dua dokumen hukum secara mendetail.

Fitur:
  1. Similarity Score antar pasal
  2. Gap Analysis otomatis
  3. Search pasal berdasarkan topik
  4. Ringkasan perbedaan (Summarization)

Cara pakai:
    from legal_comparator import LegalComparator
    model = LegalComparator()
    model.load('path/to/corpus.json')
    hasil = model.compare(doc_a='UNCAC', doc_b='UU_31_1999')
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Data Classes
# ─────────────────────────────────────────────

@dataclass
class Pasal:
    """Representasi satu unit pasal/artikel hukum."""
    id         : str
    dokumen    : str
    nomor      : str
    isi        : str
    bahasa     : str
    embedding  : Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class HasilPerbandingan:
    """Hasil lengkap perbandingan dua dokumen hukum."""
    doc_a              : str
    doc_b              : str
    similarity_matrix  : np.ndarray
    gap_results        : list
    ringkasan          : dict
    pasangan_terbaik   : list


@dataclass
class HasilSearch:
    """Hasil pencarian pasal berdasarkan topik/query."""
    query   : str
    hasil   : list          # list of (pasal, score)
    dokumen : Optional[str] = None


# ─────────────────────────────────────────────
# Main Model Class
# ─────────────────────────────────────────────

class LegalComparator:
    """
    Model utama untuk perbandingan dokumen hukum.

    Parameters
    ----------
    embedding_model : str
        Nama model sentence-transformer yang digunakan.
        Default: 'sentence-transformers/LaBSE'
        (mendukung cross-lingual EN ↔ ID)
    threshold_tinggi : float
        Batas similarity untuk kategori 'Diadopsi Penuh'. Default 0.80
    threshold_sedang : float
        Batas similarity untuk kategori 'Diadopsi Sebagian'. Default 0.65
    """

    MODEL_DEFAULT    = 'sentence-transformers/LaBSE'
    VERSI            = '1.0.0'

    def __init__(
        self,
        embedding_model  : str   = MODEL_DEFAULT,
        threshold_tinggi : float = 0.80,
        threshold_sedang : float = 0.65
    ):
        self.embedding_model_name = embedding_model
        self.threshold_tinggi     = threshold_tinggi
        self.threshold_sedang     = threshold_sedang

        self._encoder      = None   # lazy load
        self._summarizer   = None   # lazy load
        self._kw_model     = None   # lazy load

        self.corpus        : dict[str, list[Pasal]] = {}
        self.semua_pasal   : list[Pasal]             = []
        self._embeddings   : Optional[np.ndarray]    = None

        print(f'✅ LegalComparator v{self.VERSI} diinisialisasi')
        print(f'   Model embedding : {embedding_model}')
        print(f'   Threshold penuh : {threshold_tinggi}')
        print(f'   Threshold sedang: {threshold_sedang}')


    # ── PRIVATE: Lazy Load Models ──────────────────────────────

    def _get_encoder(self):
        """Load sentence encoder (LaBSE) hanya saat dibutuhkan."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            print(f'\n🔄 Memuat model embedding: {self.embedding_model_name}')
            self._encoder = SentenceTransformer(self.embedding_model_name)
            print('   ✅ Encoder siap')
        return self._encoder

    def _get_summarizer(self):
        """Load summarization model hanya saat dibutuhkan."""
        if self._summarizer is None:
            from transformers import pipeline
            print('\n🔄 Memuat model summarizer (multilingual)...')
            # Gunakan model multilingual yang ringan
            self._summarizer = pipeline(
                'summarization',
                model='facebook/mbart-large-cc25',
                tokenizer='facebook/mbart-large-cc25',
                max_length=150,
                min_length=30,
                do_sample=False
            )
            print('   ✅ Summarizer siap')
        return self._summarizer

    def _get_keyword_model(self):
        """Load KeyBERT model hanya saat dibutuhkan."""
        if self._kw_model is None:
            from keybert import KeyBERT
            print('\n🔄 Memuat KeyBERT...')
            self._kw_model = KeyBERT(
                model='paraphrase-multilingual-MiniLM-L12-v2'
            )
            print('   ✅ KeyBERT siap')
        return self._kw_model


    # ── LOAD: Muat Corpus dari JSON ────────────────────────────

    def load(self, corpus_path: str, embedding_path: Optional[str] = None):
        """
        Muat corpus hukum dari file JSON hasil preprocessing.

        Parameters
        ----------
        corpus_path : str
            Path ke corpus_structured.json
        embedding_path : str, optional
            Path ke embeddings_labse.npy (jika sudah ada)
            Jika None, embedding akan dihitung ulang.
        """
        print(f'\n📂 Memuat corpus dari: {corpus_path}')

        with open(corpus_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Bangun objek Pasal
        self.semua_pasal = []
        self.corpus      = {}

        for item in data.get('corpus', []):
            pasal = Pasal(
                id      = item['id'],
                dokumen = item['dokumen'],
                nomor   = item['pasal'],
                isi     = item['isi'],
                bahasa  = item['bahasa']
            )
            self.semua_pasal.append(pasal)
            if pasal.dokumen not in self.corpus:
                self.corpus[pasal.dokumen] = []
            self.corpus[pasal.dokumen].append(pasal)

        print(f'   ✅ {len(self.semua_pasal)} pasal dimuat')
        print(f'   📚 Dokumen: {list(self.corpus.keys())}')

        # Muat atau hitung embedding
        if embedding_path and Path(embedding_path).exists():
            print(f'\n📂 Memuat embedding dari: {embedding_path}')
            self._embeddings = np.load(embedding_path)
            # Pasangkan embedding ke setiap pasal
            for i, pasal in enumerate(self.semua_pasal):
                if i < len(self._embeddings):
                    pasal.embedding = self._embeddings[i]
            print(f'   ✅ {len(self._embeddings)} embedding dimuat')
        else:
            print('\n⚙️  Menghitung embedding (ini mungkin butuh beberapa menit)...')
            self._hitung_semua_embedding()

        return self


    # ── EMBEDDING ──────────────────────────────────────────────

    def _hitung_semua_embedding(self):
        """Hitung embedding untuk semua pasal dalam corpus."""
        encoder = self._get_encoder()
        teks    = [p.isi for p in self.semua_pasal]

        self._embeddings = encoder.encode(
            teks,
            batch_size=16,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for i, pasal in enumerate(self.semua_pasal):
            pasal.embedding = self._embeddings[i]

        print(f'   ✅ {len(self._embeddings)} embedding selesai dihitung')

    def save_embeddings(self, path: str = 'embeddings_labse.npy'):
        """Simpan embedding ke file untuk digunakan ulang."""
        if self._embeddings is not None:
            np.save(path, self._embeddings)
            print(f'💾 Embedding disimpan ke: {path}')


    # ── FITUR 1: SIMILARITY SCORE ─────────────────────────────

    def similarity_score(
        self,
        pasal_a_id : str,
        pasal_b_id : str
    ) -> dict:
        """
        Hitung similarity score antara dua pasal spesifik.

        Parameters
        ----------
        pasal_a_id : str   contoh: 'UNCAC_article_15'
        pasal_b_id : str   contoh: 'UU_31_1999_pasal_5'

        Returns
        -------
        dict dengan score, interpretasi, dan preview teks
        """
        from sklearn.metrics.pairwise import cosine_similarity

        pasal_a = self._cari_pasal_by_id(pasal_a_id)
        pasal_b = self._cari_pasal_by_id(pasal_b_id)

        if pasal_a is None:
            raise ValueError(f'Pasal tidak ditemukan: {pasal_a_id}')
        if pasal_b is None:
            raise ValueError(f'Pasal tidak ditemukan: {pasal_b_id}')

        # Hitung cosine similarity
        embed_a = pasal_a.embedding.reshape(1, -1)
        embed_b = pasal_b.embedding.reshape(1, -1)
        score   = float(cosine_similarity(embed_a, embed_b)[0][0])

        # Interpretasi
        if score >= self.threshold_tinggi:
            label = 'Sangat Mirip — Kemungkinan Besar Diadopsi'
            level = 'tinggi'
        elif score >= self.threshold_sedang:
            label = 'Cukup Mirip — Sebagian Diadopsi'
            level = 'sedang'
        elif score >= 0.45:
            label = 'Sedikit Mirip — Topik Berkaitan'
            level = 'rendah'
        else:
            label = 'Tidak Mirip — Topik Berbeda'
            level = 'sangat_rendah'

        return {
            'pasal_a'       : pasal_a_id,
            'pasal_b'       : pasal_b_id,
            'score'         : round(score, 4),
            'level'         : level,
            'interpretasi'  : label,
            'preview_a'     : pasal_a.isi[:200] + '...',
            'preview_b'     : pasal_b.isi[:200] + '...',
            'bahasa_a'      : pasal_a.bahasa,
            'bahasa_b'      : pasal_b.bahasa,
        }


    # ── FITUR 2: COMPARE (Gap Analysis) ───────────────────────

    def compare(
        self,
        doc_a          : str,
        doc_b          : str,
        top_n_gap      : int = 10
    ) -> HasilPerbandingan:
        """
        Bandingkan dua dokumen hukum secara menyeluruh.
        Menghasilkan similarity matrix + gap analysis lengkap.

        Parameters
        ----------
        doc_a      : str   Label dokumen A (contoh: 'UNCAC')
        doc_b      : str   Label dokumen B (contoh: 'UU_31_1999')
        top_n_gap  : int   Jumlah gap terbesar yang ditampilkan

        Returns
        -------
        HasilPerbandingan object
        """
        from sklearn.metrics.pairwise import cosine_similarity

        self._validasi_dokumen(doc_a)
        self._validasi_dokumen(doc_b)

        pasal_a = self.corpus[doc_a]
        pasal_b = self.corpus[doc_b]

        print(f'\n🔍 Membandingkan: {doc_a} vs {doc_b}')
        print(f'   {doc_a}: {len(pasal_a)} pasal')
        print(f'   {doc_b}: {len(pasal_b)} pasal')

        # Bangun similarity matrix
        embed_a = np.array([p.embedding for p in pasal_a])
        embed_b = np.array([p.embedding for p in pasal_b])
        sim_mat = cosine_similarity(embed_a, embed_b)

        # Gap analysis per pasal doc_a
        gap_results     = []
        pasangan_terbaik = []

        for i, pa in enumerate(pasal_a):
            skor_row    = sim_mat[i]
            best_j      = int(np.argmax(skor_row))
            best_score  = float(skor_row[best_j])
            best_pasal  = pasal_b[best_j]

            # Status adopsi
            if best_score >= self.threshold_tinggi:
                status = 'Diadopsi Penuh'
                emoji  = '✅'
            elif best_score >= self.threshold_sedang:
                status = 'Diadopsi Sebagian'
                emoji  = '🟡'
            else:
                status = 'GAP — Belum Diadopsi'
                emoji  = '❌'

            entry = {
                'pasal_a'       : f'{doc_a} Pasal {pa.nomor}',
                'pasal_a_id'    : pa.id,
                'pasal_b'       : f'{doc_b} Pasal {best_pasal.nomor}',
                'pasal_b_id'    : best_pasal.id,
                'similarity'    : round(best_score, 4),
                'status'        : status,
                'emoji'         : emoji,
                'preview_a'     : pa.isi[:150] + '...',
                'preview_b'     : best_pasal.isi[:150] + '...',
            }
            gap_results.append(entry)

            if best_score >= self.threshold_sedang:
                pasangan_terbaik.append(entry)

        # Hitung ringkasan
        n_penuh    = sum(1 for g in gap_results if g['status'] == 'Diadopsi Penuh')
        n_sebagian = sum(1 for g in gap_results if g['status'] == 'Diadopsi Sebagian')
        n_gap      = sum(1 for g in gap_results if g['status'] == 'GAP — Belum Diadopsi')
        total      = len(gap_results)

        ringkasan = {
            'doc_a'              : doc_a,
            'doc_b'              : doc_b,
            'total_pasal_a'      : total,
            'diadopsi_penuh'     : n_penuh,
            'diadopsi_sebagian'  : n_sebagian,
            'gap_belum_diadopsi' : n_gap,
            'persen_penuh'       : round(n_penuh / total * 100, 1) if total else 0,
            'persen_sebagian'    : round(n_sebagian / total * 100, 1) if total else 0,
            'persen_gap'         : round(n_gap / total * 100, 1) if total else 0,
            'rata_rata_similarity': round(float(np.mean([g['similarity'] for g in gap_results])), 4),
            'gap_terbesar'       : sorted(gap_results, key=lambda x: x['similarity'])[:top_n_gap]
        }

        self._cetak_ringkasan(ringkasan)

        return HasilPerbandingan(
            doc_a             = doc_a,
            doc_b             = doc_b,
            similarity_matrix = sim_mat,
            gap_results       = gap_results,
            ringkasan         = ringkasan,
            pasangan_terbaik  = pasangan_terbaik
        )


    # ── FITUR 3: SEARCH ───────────────────────────────────────

    def search(
        self,
        query    : str,
        dokumen  : Optional[str] = None,
        top_n    : int = 5
    ) -> HasilSearch:
        """
        Cari pasal yang paling relevan dengan query/topik tertentu.
        Mendukung pencarian dalam bahasa Indonesia maupun Inggris.

        Parameters
        ----------
        query   : str   Topik atau pertanyaan (contoh: 'suap pejabat negara')
        dokumen : str   Filter ke dokumen tertentu (opsional)
        top_n   : int   Jumlah hasil yang dikembalikan

        Returns
        -------
        HasilSearch object
        """
        from sklearn.metrics.pairwise import cosine_similarity

        encoder      = self._get_encoder()
        query_embed  = encoder.encode([query], normalize_embeddings=True)

        # Filter dokumen jika diminta
        if dokumen:
            self._validasi_dokumen(dokumen)
            pool = self.corpus[dokumen]
        else:
            pool = self.semua_pasal

        # Hitung similarity query vs semua pasal
        pool_embed = np.array([p.embedding for p in pool])
        scores     = cosine_similarity(query_embed, pool_embed)[0]

        # Urutkan & ambil top_n
        top_idx = np.argsort(scores)[::-1][:top_n]

        hasil = []
        for idx in top_idx:
            pasal = pool[idx]
            hasil.append({
                'rank'       : len(hasil) + 1,
                'pasal_id'   : pasal.id,
                'dokumen'    : pasal.dokumen,
                'nomor_pasal': pasal.nomor,
                'score'      : round(float(scores[idx]), 4),
                'bahasa'     : pasal.bahasa,
                'preview'    : pasal.isi[:300] + '...'
            })

        print(f'\n🔎 Hasil pencarian: "{query}"')
        print(f'   Dokumen filter: {dokumen or "Semua"}')
        print(f'   {"─" * 55}')
        for h in hasil:
            print(f'   #{h["rank"]} [{h["score"]:.3f}] {h["dokumen"]} Pasal {h["nomor_pasal"]}')
            print(f'        {h["preview"][:100]}...')
            print()

        return HasilSearch(query=query, hasil=hasil, dokumen=dokumen)


    # ── FITUR 4: SUMMARIZE ────────────────────────────────────

    def summarize_gap(
        self,
        hasil_compare  : HasilPerbandingan,
        top_n_gap      : int = 5,
        gaya           : str = 'naratif'   # 'naratif' | 'poin' | 'tabel'
    ) -> str:
        """
        Buat ringkasan perbedaan antara dua dokumen hukum
        berdasarkan hasil compare().

        Parameters
        ----------
        hasil_compare : HasilPerbandingan   output dari compare()
        top_n_gap     : int                 berapa gap yang dibahas
        gaya          : str                 format output ringkasan

        Returns
        -------
        str   ringkasan dalam format yang dipilih
        """
        r  = hasil_compare.ringkasan
        ga = r['gap_terbesar'][:top_n_gap]

        if gaya == 'tabel':
            return self._ringkasan_tabel(r, ga)
        elif gaya == 'poin':
            return self._ringkasan_poin(r, ga)
        else:
            return self._ringkasan_naratif(r, ga)

    def _ringkasan_naratif(self, r: dict, gap_list: list) -> str:
        lines = [
            '=' * 65,
            f'RINGKASAN PERBANDINGAN: {r["doc_a"]} vs {r["doc_b"]}',
            '=' * 65,
            '',
            f'Dari total {r["total_pasal_a"]} pasal/artikel dalam {r["doc_a"]},',
            f'perbandingan dengan {r["doc_b"]} menunjukkan bahwa:',
            '',
            f'  • {r["diadopsi_penuh"]} pasal ({r["persen_penuh"]}%) telah diadopsi secara penuh',
            f'  • {r["diadopsi_sebagian"]} pasal ({r["persen_sebagian"]}%) diadopsi sebagian',
            f'  • {r["gap_belum_diadopsi"]} pasal ({r["persen_gap"]}%) belum diadopsi (GAP)',
            '',
            f'Rata-rata kemiripan semantik: {r["rata_rata_similarity"]:.4f}',
            '',
            f'PASAL DENGAN GAP TERBESAR (belum diadopsi):',
            '-' * 65,
        ]
        for i, g in enumerate(gap_list, 1):
            lines.append(f'{i}. {g["pasal_a"]}')
            lines.append(f'   Padanan terdekat : {g["pasal_b"]}')
            lines.append(f'   Similarity score : {g["similarity"]:.4f}')
            lines.append(f'   Status           : {g["emoji"]} {g["status"]}')
            lines.append(f'   Isi              : {g["preview_a"][:120]}...')
            lines.append('')
        lines.append('=' * 65)
        return '\n'.join(lines)

    def _ringkasan_poin(self, r: dict, gap_list: list) -> str:
        lines = [
            f'# Perbandingan {r["doc_a"]} vs {r["doc_b"]}',
            '',
            '## Statistik',
            f'- Total pasal dianalisis : {r["total_pasal_a"]}',
            f'- ✅ Diadopsi penuh      : {r["diadopsi_penuh"]} ({r["persen_penuh"]}%)',
            f'- 🟡 Diadopsi sebagian   : {r["diadopsi_sebagian"]} ({r["persen_sebagian"]}%)',
            f'- ❌ GAP/belum diadopsi  : {r["gap_belum_diadopsi"]} ({r["persen_gap"]}%)',
            '',
            '## Gap Terbesar',
        ]
        for g in gap_list:
            lines.append(f'- {g["emoji"]} **{g["pasal_a"]}** (score: {g["similarity"]:.3f})')
            lines.append(f'  → Padanan: {g["pasal_b"]}')
        return '\n'.join(lines)

    def _ringkasan_tabel(self, r: dict, gap_list: list) -> str:
        header = f'{"Pasal A":<25} {"Padanan B":<30} {"Score":>7} {"Status"}'
        sep    = '-' * 80
        rows   = [header, sep]
        for g in gap_list:
            rows.append(
                f'{g["pasal_a"]:<25} {g["pasal_b"]:<30} '
                f'{g["similarity"]:>7.4f} {g["emoji"]} {g["status"]}'
            )
        return '\n'.join(rows)


    # ── UTILITAS INTERNAL ──────────────────────────────────────

    def _cari_pasal_by_id(self, pasal_id: str) -> Optional[Pasal]:
        for p in self.semua_pasal:
            if p.id == pasal_id:
                return p
        return None

    def _validasi_dokumen(self, label: str):
        if label not in self.corpus:
            tersedia = list(self.corpus.keys())
            raise ValueError(
                f'Dokumen "{label}" tidak ditemukan.\n'
                f'Dokumen tersedia: {tersedia}'
            )

    def _cetak_ringkasan(self, r: dict):
        print(f'\n📊 RINGKASAN PERBANDINGAN')
        print(f'   {"─" * 45}')
        print(f'   ✅ Diadopsi Penuh    : {r["diadopsi_penuh"]:>3} pasal ({r["persen_penuh"]}%)')
        print(f'   🟡 Diadopsi Sebagian : {r["diadopsi_sebagian"]:>3} pasal ({r["persen_sebagian"]}%)')
        print(f'   ❌ GAP               : {r["gap_belum_diadopsi"]:>3} pasal ({r["persen_gap"]}%)')
        print(f'   📐 Avg Similarity    : {r["rata_rata_similarity"]:.4f}')

    def daftar_dokumen(self) -> list:
        """Tampilkan semua dokumen yang tersedia dalam corpus."""
        return list(self.corpus.keys())

    def info_dokumen(self, label: str) -> dict:
        """Tampilkan informasi detail satu dokumen."""
        self._validasi_dokumen(label)
        pasal_list = self.corpus[label]
        return {
            'label'       : label,
            'jumlah_pasal': len(pasal_list),
            'bahasa'      : pasal_list[0].bahasa if pasal_list else '-',
            'id_pasal'    : [p.id for p in pasal_list]
        }
