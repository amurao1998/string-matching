import pandas as pd, numpy as np, re, sparse_dot_topn.sparse_dot_topn as ct
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import time

def ngrams(string, n=2):
    string = (re.sub(r'[,-./]|\sBD',r'', string)).upper()
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(sparse_matrix, A, B, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = A[sparserows[index]]
        right_side[index] = B[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame({'left_side': left_side,
                         'right_side': right_side,
                         'similairity': similairity})

df_dirty = {"name":["eng","english","mra","mr","zulu"] }

df_clean = {"name":["ab","aa","af","sq","am","ar","hy","as","ay","az","ba","eu",
"bn","dz","bh","bi","br","bg","my","be","km","ca","zh","co","hr","cs","da","nl","en",
"eo","et","fo","fj","fi","fr","fy","gd","gl","ka","de","el","kl","gn","gu","ha","iw",
"hi","hu","is","in","ia","ie","ik","ga","it","ja","jw","kn","ks","kk","rw","ky","rn",
"ko","ku","lo","la","lv","ln","lt","mk","mg","ms","ml","mt","mi","mr","mo","mn","na",
"ne","no","oc","or","om","ps","fa","pl","pt","pa","qu","rm","ro","ru","sm","sg","sa",
"sr","sh","st","tn","sn","sd","si","ss","sk","sl","so","es","su","sw","sv","tl","tg",
"ta","tt","tt","th","bo","ti","to","ts","tr","tk","tw","uk","ur","uz","vi","vo","cy",
"wo","xh","ji","yo","zu"]}
vectorizer = TfidfVectorizer(analyzer=ngrams)
tf_idf_matrix_clean = vectorizer.fit_transform(df_clean['name'])
tf_idf_matrix_dirty = vectorizer.transform(df_dirty["name"])

t1 = time.time()
matches = awesome_cossim_top(tf_idf_matrix_dirty, tf_idf_matrix_clean.transpose(), 1, 0)
t = time.time()-t1
print("SELFTIMED:", t)

matches_df = get_matches_df(matches, df_dirty["name"], df_clean['name'], top = len(df_dirty["name"]))

print(matches_df.size)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(matches_df)