"""Legacy initial alphabets and declared capacity grid."""
import string
SIZES = [64, 80, 96, 112, 128, 160, 192, 200, 224, 240, 256, 272, 288, 300, 320, 352, 384, 416, 448, 450, 480, 512]

def alphabet_protocols(train):
    old = set(range(32, 127)) | {9, 10, 13}
    # Fixed character-level protocols, no complete command/variable composites.
    core = set((string.ascii_lowercase+string.digits+'\\{}()[]^_+-=*/., \t\n\r<>|:;!&').encode())
    return {
        'core': dict(alphabet=sorted(core), rule='fixed lower-case Latin, digits, LaTeX punctuation and whitespace'),
        'letters': dict(alphabet=sorted(core | set(string.ascii_uppercase.encode())), rule='core plus all upper-case Latin'),
        'latex_ascii': dict(alphabet=sorted(old-set(b'"#$%?@`')), rule='ASCII/whitespace excluding double quote, hash, dollar, percent, question mark, at sign, backtick'),
        'ascii95': dict(alphabet=sorted(old-set(b'"#$')), rule='ASCII/whitespace excluding double quote, hash, dollar'),
        'ascii98': dict(alphabet=sorted(old), rule='historical 95 printable ASCII plus TAB/LF/CR'),
        'train_observed': dict(alphabet=sorted(set(b'0123456789') | set().union(*(set(r['latex'].encode()) for r in train))),
                               rule='train-only observed bytes plus all digits; diagnostic adaptive arm, not a fixed protocol'),
    }
