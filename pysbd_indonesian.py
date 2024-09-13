import pysbd
from pysbd.abbreviation_replacer import AbbreviationReplacer
from pysbd.lang.common import Common, Standard

'''
pySBD Indonesian class based on InaNLP.
'''


# Copied exactly from
# https://github.com/panggi/pujangga/tree/master/resource/sentencedetector/acronym.txt:
ID_ABBR = [
    'a.d.', 'a.d.c.', 'A.H.', 'a.i.', 'a.l.', 'a.m.',
    'A.M.', 'A.M.v.B.', 'a.n.', 'a.n.b.', 'a.p.', 'a.s.',
    'adm.', 'A.', 'Ak.', 'Akt.', 'Alm.', 'alm.',
    'Almh.', 'almh.', 'Anm.', 'Ant.', 'art.', 'Ass.',
    'ass.', 'ay.', 'B.', 'B.A.', 'B.B.A.', 'B.Ch.E.',
    'b.d.', 'B.J.', 'B.Sc.', 'Bc.A.K.', 'Bc.Ac.P.', 'Bc.Hk.',
    'Bc.I.', 'Bc.KWN.', 'Bc.T.T.', 'C.', 'C.C.', 'C.O.',
    'c.q.', 'cf.', 'Co.', 'Corp.', 'D.', 'd.a.',
    'd.l.', 'D.Sc.', 'dkk.', 'dll.', 'dr.', 'Dr.',
    'Dra.', 'drg.', 'drh.', 'Drs.', 'dsb.', 'dst.',
    'e.g.', 'E.Z.', 'ed.', 'eks.', 'et al.', 'et seq.',
    'etc.', 'ext.', 'E.', 'fol.', 'ft.', 'F.',
    'Fr.', 'Fv.', 'G.', 'Gg.', 'H.', 'H.C.',
    'Hj.', 'hlm.', 'i.c.', 'i.e.', 'ibid.', 'id.',
    'I.', 'Inc.', 'Ir.', 'It.', 'J.', 'Jend.',
    'Jl.', 'Jln.', 'K.', 'K.', 'K.H.', 'k.l.',
    'kab.', 'kapt.', 'kec.', 'kel.', 'kep.', 'l.l.',
    'lamp.', 'L.', 'Litt.', 'LL.B.', 'LL.D.', 'loc. cit.',
    'log.', 'Ltd.', 'ltd.', 'M.', 'M.A.', 'M.Ag.',
    'M.B.A.', 'M.D.', 'M.Hum.', 'M.Kes.', 'M.Kom.', 'M.M.',
    'M.P.', 'M.P.A.', 'M.P.H.', 'M.Pd.', 'M.Ph.', 'M.Sc.',
    'M.Si.', 'M.Sn.', 'M.T.', 'mgr.', 'Moh.', 'Muh.',
    'm.p.h.', 'Mr.', 'Mrs.', 'n.b.', 'N.', 'N.B.',
    'N.N.', 'Nn.', 'No.', 'no.', 'Ny.', 'O.',
    'op. cit.', 'p.a.', 'p.c.', 'p.f.', 'p.f.v.', 'p.m.',
    'p.p.', 'p.r.', 'P.', 'Ph.D.', 'Phil.', 'Pjs.',
    'Plh.', 'Ppn.', 'Prof.', 'Psi.', 'psw.', 'q.e.',
    'q.q.', 'q.v.', 'Q.', 'R.', 'R.A.', 'R.Ay.',
    'R.M.', 'r.p.m.', 'r.p.s.', 'R.R.', 'red.', 'ref.',
    'reg.', 'rhs.', 'S.', 'S.Ag.', 's.d.', 'S.E.',
    'S.H.', 'S.Hut.', 'S.IP.', 'S.Ked.', 'S.Kedg.', 'S.Kedh.',
    'S.KG.', 'S.KH.', 'S.KM.', 'S.Kom.', 'S.M.', 'S.P.',
    'S.Pd.', 'S.Pi.', 'S.Pol.', 'S.Psi.', 'S.Pt.', 'S.S.',
    'S.Si.', 'S.Sn.', 'S.Sos.', 'S.T.', 'S.Tekp.', 'S.Th.',
    'S.TP.', 'S.V.P.', 'saw.', 'sbb.', 'sbg.', 'ssk.',
    'St.', 'swt.', 'T.', 't.t.', 'Tap.', 'Tb.',
    'Tbk.', 'tel.', 'Th.', 'tsb.', 'tst.', 'ttd.',
    'ttg.', 'U.', 'u.b.', 'u.p.', 'V.', 'V.O.C.',
    'v.v.', 'vol.', 'W.', 'X.', 'y.l.', 'ybs.',
    'Y.', 'Ytc.', 'Yth.', 'Yts.', 'Z.'
    ]
# PySBD requires lowercased and without final dot:
ID_ABBR_PYSBD = [a.lower().removesuffix('.') for a in ID_ABBR]


class Indonesian(Common, Standard):

    # Based on https://github.com/panggi/pujangga/blob/master/resource/sentencedetector/
    # delimiter.txt
    SENTENCE_BOUNDARY_REGEX = r'.*?[.!?]|.*?$'
    Punctuations = ['.', '!', '?']

    class AbbreviationReplacer(AbbreviationReplacer):
        SENTENCE_STARTERS = []

    class Abbreviation(Standard.Abbreviation):
        # PySBD: Abbreviations need to be "prepositive" in order to work before
        # capitalized words, e.g. 'Dr. Alexander':
        ABBREVIATIONS = ID_ABBR_PYSBD
        PREPOSITIVE_ABBREVIATIONS = ID_ABBR_PYSBD
        NUMBER_ABBREVIATIONS = []


if 'id' in pysbd.languages.LANGUAGE_CODES:
    raise Exception(
        'It seems that the installed version of the the pysbd package already contains '
        'Indonesian ("id"). Using pysbd_indonesian would override it.'
        )

pysbd.languages.LANGUAGE_CODES['id'] = Indonesian


if __name__ == '__main__':
    sseg = pysbd.Segmenter('id').segment
    # Not realy a test case, just an example (sentences from Wikipedia):
    example = (
        'H. Ahmad Muzani, S.Sos. (lahir 15 Juli 1968 [1]) adalah seorang Pengusaha dan '
        'Politisi Indonesia dari Partai Gerakan Indonesia Raya (Gerindra). '
        'Dr. Alexander Andries Maramis atau lebih dikenal. '
        'Rey sebuah lukisan yang menggambarkan diri sang dokter pada tahun 1889, '
        'tetapi Dr Rey tidak menyukai lukisan tersebut sehingga ia memakainya untuk '
        'memperbaiki kandang ayam dan lalu memberikannya kepada orang lain? '
        'Pada tahun 2016, potret tersebut disimpan di Museum Seni Rupa Murni Pushkin '
        'dan nilainya diperkirakan melebihi $50 juta! '
        'Indonesia merupakan negara terluas ke-14 sekaligus negara kepulauan terbesar '
        'di dunia dengan luas wilayah sebesar 1.904.569 km², serta negara dengan pulau '
        'terbanyak ke-6 di dunia, dengan jumlah 17.504 pulau.'
        )

    for s in sseg(example):
        print('*', s)
