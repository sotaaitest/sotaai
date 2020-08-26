import nlp
from transformers import *
import transformers
from sotaai.nlp.hugface_models import mod_list

datasets_list = nlp.list_datasets()

datasets = [dataset.id for dataset in datasets_list]

DATASETS = {
    'ai2_arc':['ARC-Challenge', 'ARC-Easy'],
    'biomrc':['biomrc_large_A', 'biomrc_large_B', 'biomrc_small_A', 'biomrc_small_B', 'biomrc_tiny_A', 'biomrc_tiny_B'],
    #'blended_skill_talk':, ---using custom config default?
    'blimp': ['adjunct_island', 'anaphor_gender_agreement', 'anaphor_number_agreement', 'animate_subject_passive', 'animate_subject_trans', 'causative', 'complex_NP_island', 'coordinate_structure_constraint_complex_left_branch', 'coordinate_structure_constraint_object_extraction', 'determiner_noun_agreement_1', 'determiner_noun_agreement_2', 'determiner_noun_agreement_irregular_1', 'determiner_noun_agreement_irregular_2', 'determiner_noun_agreement_with_adj_2', 'determiner_noun_agreement_with_adj_irregular_1', 'determiner_noun_agreement_with_adj_irregular_2', 'determiner_noun_agreement_with_adjective_1', 'distractor_agreement_relational_noun', 'distractor_agreement_relative_clause', 'drop_argument', 'ellipsis_n_bar_1', 'ellipsis_n_bar_2', 'existential_there_object_raising', 'existential_there_quantifiers_1', 'existential_there_quantifiers_2', 'existential_there_subject_raising', 'expletive_it_object_raising', 'inchoative', 'intransitive', 'irregular_past_participle_adjectives', 'irregular_past_participle_verbs', 'irregular_plural_subject_verb_agreement_1', 'irregular_plural_subject_verb_agreement_2', 'left_branch_island_echo_question', 'left_branch_island_simple_question', 'matrix_question_npi_licensor_present', 'npi_present_1', 'npi_present_2', 'only_npi_licensor_present', 'only_npi_scope', 'passive_1', 'passive_2', 'principle_A_c_command', 'principle_A_case_1', 'principle_A_case_2', 'principle_A_domain_1', 'principle_A_domain_2', 'principle_A_domain_3', 'principle_A_reconstruction', 'regular_plural_subject_verb_agreement_1', 'regular_plural_subject_verb_agreement_2', 'sentential_negation_npi_licensor_present', 'sentential_negation_npi_scope', 'sentential_subject_island', 'superlative_quantifiers_1', 'superlative_quantifiers_2', 'tough_vs_raising_1', 'tough_vs_raising_2', 'transitive', 'wh_island', 'wh_questions_object_gap', 'wh_questions_subject_gap', 'wh_questions_subject_gap_long_distance', 'wh_vs_that_no_gap', 'wh_vs_that_no_gap_long_distance', 'wh_vs_that_with_gap', 'wh_vs_that_with_gap_long_distance'],
    'break_data':['QDMR-high-level', 'QDMR-high-level-lexicon', 'QDMR', 'QDMR-lexicon', 'logical-forms'],
    'c4':['en', 'en.noclean', 'en.realnewslike', 'en.webtextlike'],
    'cfq':['mcd1', 'mcd2', 'mcd3', 'question_complexity_split', 'question_pattern_split', 'query_complexity_split', 'query_pattern_split', 'random_split'],
    'cnn_dailymail':['3.0.0', '1.0.0', '2.0.0'],
    'compguesswhat':['compguesswhat-original', 'compguesswhat-zero_shot'],
    'cos_e':['v1.0', 'v1.11'],
    'discofuse':['discofuse-sport', 'discofuse-wikipedia'],
    'fever':['v1.0', 'v2.0', 'wiki_pages'],
    'flores':['neen', 'sien'],
    'glue':['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli', 'ax'],
    'hansards':['house', 'senate'],
    'hyperpartisan_news_detection':['byarticle', 'bypublisher'],
    'kor_nli':['multi_nli', 'snli', 'xnli'],
    #'lhoestq/c4':,No module named 'tldextract'
    'lince':['lid_spaeng', 'lid_hineng', 'lid_msaea', 'lid_nepeng', 'pos_spaeng', 'pos_hineng', 'ner_spaeng', 'ner_msaea', 'ner_hineng', 'sa_spaeng'],
    'math_dataset':['algebra__linear_1d', 'algebra__linear_1d_composed', 'algebra__linear_2d', 'algebra__linear_2d_composed', 'algebra__polynomial_roots', 'algebra__polynomial_roots_composed', 'algebra__sequence_next_term', 'algebra__sequence_nth_term', 'arithmetic__add_or_sub', 'arithmetic__add_or_sub_in_base', 'arithmetic__add_sub_multiple', 'arithmetic__div', 'arithmetic__mixed', 'arithmetic__mul', 'arithmetic__mul_div_multiple', 'arithmetic__nearest_integer_root', 'arithmetic__simplify_surd', 'calculus__differentiate', 'calculus__differentiate_composed', 'comparison__closest', 'comparison__closest_composed', 'comparison__kth_biggest', 'comparison__kth_biggest_composed', 'comparison__pair', 'comparison__pair_composed', 'comparison__sort', 'comparison__sort_composed', 'measurement__conversion', 'measurement__time', 'numbers__base_conversion', 'numbers__div_remainder', 'numbers__div_remainder_composed', 'numbers__gcd', 'numbers__gcd_composed', 'numbers__is_factor', 'numbers__is_factor_composed', 'numbers__is_prime', 'numbers__is_prime_composed', 'numbers__lcm', 'numbers__lcm_composed', 'numbers__list_prime_factors', 'numbers__list_prime_factors_composed', 'numbers__place_value', 'numbers__place_value_composed', 'numbers__round_number', 'numbers__round_number_composed', 'polynomials__add', 'polynomials__coefficient_named', 'polynomials__collect', 'polynomials__compose', 'polynomials__evaluate', 'polynomials__evaluate_composed', 'polynomials__expand', 'polynomials__simplify_power', 'probability__swr_p_level_set', 'probability__swr_p_sequence'],
    'mlqa':['mlqa-translate-train.ar', 'mlqa-translate-train.de', 'mlqa-translate-train.vi', 'mlqa-translate-train.zh', 'mlqa-translate-train.es', 'mlqa-translate-train.hi', 'mlqa-translate-test.ar', 'mlqa-translate-test.de', 'mlqa-translate-test.vi', 'mlqa-translate-test.zh', 'mlqa-translate-test.es', 'mlqa-translate-test.hi', 'mlqa.ar.ar', 'mlqa.ar.de', 'mlqa.ar.vi', 'mlqa.ar.zh', 'mlqa.ar.en', 'mlqa.ar.es', 'mlqa.ar.hi', 'mlqa.de.ar', 'mlqa.de.de', 'mlqa.de.vi', 'mlqa.de.zh', 'mlqa.de.en', 'mlqa.de.es', 'mlqa.de.hi', 'mlqa.vi.ar', 'mlqa.vi.de', 'mlqa.vi.vi', 'mlqa.vi.zh', 'mlqa.vi.en', 'mlqa.vi.es', 'mlqa.vi.hi', 'mlqa.zh.ar', 'mlqa.zh.de', 'mlqa.zh.vi', 'mlqa.zh.zh', 'mlqa.zh.en', 'mlqa.zh.es', 'mlqa.zh.hi', 'mlqa.en.ar', 'mlqa.en.de', 'mlqa.en.vi', 'mlqa.en.zh', 'mlqa.en.en', 'mlqa.en.es', 'mlqa.en.hi', 'mlqa.es.ar', 'mlqa.es.de', 'mlqa.es.vi', 'mlqa.es.zh', 'mlqa.es.en', 'mlqa.es.es', 'mlqa.es.hi', 'mlqa.hi.ar', 'mlqa.hi.de', 'mlqa.hi.vi', 'mlqa.hi.zh', 'mlqa.hi.en', 'mlqa.hi.es', 'mlqa.hi.hi'],
    #'natural_questions':, Not enough disk space. Needed: 134.92 GiB (download: 41.97 GiB, generated: 92.95 GiB)
    'newsgroup':['18828_alt.atheism', '18828_comp.graphics', '18828_comp.os.ms-windows.misc', '18828_comp.sys.ibm.pc.hardware', '18828_comp.sys.mac.hardware', '18828_comp.windows.x', '18828_misc.forsale', '18828_rec.autos', '18828_rec.motorcycles', '18828_rec.sport.baseball', '18828_rec.sport.hockey', '18828_sci.crypt', '18828_sci.electronics', '18828_sci.med', '18828_sci.space', '18828_soc.religion.christian', '18828_talk.politics.guns', '18828_talk.politics.mideast', '18828_talk.politics.misc', '18828_talk.religion.misc', '19997_alt.atheism', '19997_comp.graphics', '19997_comp.os.ms-windows.misc', '19997_comp.sys.ibm.pc.hardware', '19997_comp.sys.mac.hardware', '19997_comp.windows.x', '19997_misc.forsale', '19997_rec.autos', '19997_rec.motorcycles', '19997_rec.sport.baseball', '19997_rec.sport.hockey', '19997_sci.crypt', '19997_sci.electronics', '19997_sci.med', '19997_sci.space', '19997_soc.religion.christian', '19997_talk.politics.guns', '19997_talk.politics.mideast', '19997_talk.politics.misc', '19997_talk.religion.misc', 'bydate_alt.atheism', 'bydate_comp.graphics', 'bydate_comp.os.ms-windows.misc', 'bydate_comp.sys.ibm.pc.hardware', 'bydate_comp.sys.mac.hardware', 'bydate_comp.windows.x', 'bydate_misc.forsale', 'bydate_rec.autos', 'bydate_rec.motorcycles', 'bydate_rec.sport.baseball', 'bydate_rec.sport.hockey', 'bydate_sci.crypt', 'bydate_sci.electronics', 'bydate_sci.med', 'bydate_sci.space', 'bydate_soc.religion.christian', 'bydate_talk.politics.guns', 'bydate_talk.politics.mideast', 'bydate_talk.politics.misc', 'bydate_talk.religion.misc'],
    #'newsroom':,#Using custom data configuration default which needs manual download
    'openbookqa':['main', 'additional'],
    'para_crawl':['enbg', 'encs', 'enda', 'ende', 'enel', 'enes', 'enet', 'enfi', 'enfr', 'enga', 'enhr', 'enhu', 'enit', 'enlt', 'enlv', 'enmt', 'ennl', 'enpl', 'enpt', 'enro', 'ensk', 'ensl', 'ensv'],
    'qa4mre':['2011.main.DE', '2011.main.EN', '2011.main.ES', '2011.main.IT', '2011.main.RO', '2012.main.AR', '2012.main.BG', '2012.main.DE', '2012.main.EN', '2012.main.ES', '2012.main.IT', '2012.main.RO', '2012.alzheimers.EN', '2013.main.AR', '2013.main.BG', '2013.main.EN', '2013.main.ES', '2013.main.RO', '2013.alzheimers.EN', '2013.entrance_exam.EN'],
    'qangaroo':['medhop', 'masked_medhop', 'wikihop', 'masked_wikihop'],
    'qanta':['mode=full,char_skip=25', 'mode=first,char_skip=25', 'mode=sentences,char_skip=25', 'mode=runs,char_skip=25'],
    #'reclor':#config default requires manual instalation,
    'reddit_tifu':['short', 'long'],
    'scan':['simple', 'addprim_jump', 'addprim_turn_left', 'filler_num0', 'filler_num1', 'filler_num2', 'filler_num3', 'length', 'template_around_right', 'template_jump_around_right', 'template_opposite_right', 'template_right'],
    'scientific_papers':['pubmed', 'arxiv'],
    'scifact':['corpus', 'claims'],
    'scitail':['snli_format', 'tsv_format', 'dgem_format', 'predictor_format'],
    'search_qa':['raw_jeopardy', 'train_test_val'],
    'squad_es': ['v1.1.0', 'v2.0.0'],
    'squadshifts':['new_wiki', 'nyt', 'reddit', 'amazon'],
    'style_change_detection':['narrow', 'wide'],
    'super_glue':['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc', 'wsc.fixed', 'axb', 'axg'],
    'ted_hrlr':['az_to_en', 'aztr_to_en', 'be_to_en', 'beru_to_en', 'es_to_pt', 'fr_to_pt', 'gl_to_en', 'glpt_to_en', 'he_to_pt', 'it_to_pt', 'pt_to_en', 'ru_to_en', 'ru_to_pt', 'tr_to_en'],
    'trivia_qa':['rc', 'rc.nocontext', 'unfiltered', 'unfiltered.nocontext'],
    'tydiqa':['primary_task', 'secondary_task'],
    'ubuntu_dialogs_corpus':['train', 'dev_test'],
    'web_of_science': ['WOS5736', 'WOS11967', 'WOS46985'],
    'wiki40b':['en', 'ar', 'zh-cn', 'zh-tw', 'nl', 'fr', 'de', 'it', 'ja', 'ko', 'pl', 'pt', 'ru', 'es', 'th', 'tr', 'bg', 'ca', 'cs', 'da', 'el', 'et', 'fa', 'fi', 'he', 'hi', 'hr', 'hu', 'id', 'lt', 'lv', 'ms', 'no', 'ro', 'sk', 'sl', 'sr', 'sv', 'tl', 'uk', 'vi'],
    'wiki_dpr':['psgs_w100_with_nq_embeddings', 'psgs_w100_no_embeddings', 'dummy_psgs_w100_with_nq_embeddings', 'dummy_psgs_w100_no_embeddings'],
    'wiki_snippets':['wiki40b_en_100_0', 'wikipedia_en_100_0'],
    'wikihow':['all', 'sep'],
    'wikipedia':['20200501.aa', '20200501.ab', '20200501.ace', '20200501.ady', '20200501.af', '20200501.ak', '20200501.als', '20200501.am', '20200501.an', '20200501.ang', '20200501.ar', '20200501.arc', '20200501.arz', '20200501.as', '20200501.ast', '20200501.atj', '20200501.av', '20200501.ay', '20200501.az', '20200501.azb', '20200501.ba', '20200501.bar', '20200501.bat-smg', '20200501.bcl', '20200501.be', '20200501.be-x-old', '20200501.bg', '20200501.bh', '20200501.bi', '20200501.bjn', '20200501.bm', '20200501.bn', '20200501.bo', '20200501.bpy', '20200501.br', '20200501.bs', '20200501.bug', '20200501.bxr', '20200501.ca', '20200501.cbk-zam', '20200501.cdo', '20200501.ce', '20200501.ceb', '20200501.ch', '20200501.cho', '20200501.chr', '20200501.chy', '20200501.ckb', '20200501.co', '20200501.cr', '20200501.crh', '20200501.cs', '20200501.csb', '20200501.cu', '20200501.cv', '20200501.cy', '20200501.da', '20200501.de', '20200501.din', '20200501.diq', '20200501.dsb', '20200501.dty', '20200501.dv', '20200501.dz', '20200501.ee', '20200501.el', '20200501.eml', '20200501.en', '20200501.eo', '20200501.es', '20200501.et', '20200501.eu', '20200501.ext', '20200501.fa', '20200501.ff', '20200501.fi', '20200501.fiu-vro', '20200501.fj', '20200501.fo', '20200501.fr', '20200501.frp', '20200501.frr', '20200501.fur', '20200501.fy', '20200501.ga', '20200501.gag', '20200501.gan', '20200501.gd', '20200501.gl', '20200501.glk', '20200501.gn', '20200501.gom', '20200501.gor', '20200501.got', '20200501.gu', '20200501.gv', '20200501.ha', '20200501.hak', '20200501.haw', '20200501.he', '20200501.hi', '20200501.hif', '20200501.ho', '20200501.hr', '20200501.hsb', '20200501.ht', '20200501.hu', '20200501.hy', '20200501.ia', '20200501.id', '20200501.ie', '20200501.ig', '20200501.ii', '20200501.ik', '20200501.ilo', '20200501.inh', '20200501.io', '20200501.is', '20200501.it', '20200501.iu', '20200501.ja', '20200501.jam', '20200501.jbo', '20200501.jv', '20200501.ka', '20200501.kaa', '20200501.kab', '20200501.kbd', '20200501.kbp', '20200501.kg', '20200501.ki', '20200501.kj', '20200501.kk', '20200501.kl', '20200501.km', '20200501.kn', '20200501.ko', '20200501.koi', '20200501.krc', '20200501.ks', '20200501.ksh', '20200501.ku', '20200501.kv', '20200501.kw', '20200501.ky', '20200501.la', '20200501.lad', '20200501.lb', '20200501.lbe', '20200501.lez', '20200501.lfn', '20200501.lg', '20200501.li', '20200501.lij', '20200501.lmo', '20200501.ln', '20200501.lo', '20200501.lrc', '20200501.lt', '20200501.ltg', '20200501.lv', '20200501.mai', '20200501.map-bms', '20200501.mdf', '20200501.mg', '20200501.mh', '20200501.mhr', '20200501.mi', '20200501.min', '20200501.mk', '20200501.ml', '20200501.mn', '20200501.mr', '20200501.mrj', '20200501.ms', '20200501.mt', '20200501.mus', '20200501.mwl', '20200501.my', '20200501.myv', '20200501.mzn', '20200501.na', '20200501.nah', '20200501.nap', '20200501.nds', '20200501.nds-nl', '20200501.ne', '20200501.new', '20200501.ng', '20200501.nl', '20200501.nn', '20200501.no', '20200501.nov', '20200501.nrm', '20200501.nso', '20200501.nv', '20200501.ny', '20200501.oc', '20200501.olo', '20200501.om', '20200501.or', '20200501.os', '20200501.pa', '20200501.pag', '20200501.pam', '20200501.pap', '20200501.pcd', '20200501.pdc', '20200501.pfl', '20200501.pi', '20200501.pih', '20200501.pl', '20200501.pms', '20200501.pnb', '20200501.pnt', '20200501.ps', '20200501.pt', '20200501.qu', '20200501.rm', '20200501.rmy', '20200501.rn', '20200501.ro', '20200501.roa-rup', '20200501.roa-tara', '20200501.ru', '20200501.rue', '20200501.rw', '20200501.sa', '20200501.sah', '20200501.sat', '20200501.sc', '20200501.scn', '20200501.sco', '20200501.sd', '20200501.se', '20200501.sg', '20200501.sh', '20200501.si', '20200501.simple', '20200501.sk', '20200501.sl', '20200501.sm', '20200501.sn', '20200501.so', '20200501.sq', '20200501.sr', '20200501.srn', '20200501.ss', '20200501.st', '20200501.stq', '20200501.su', '20200501.sv', '20200501.sw', '20200501.szl', '20200501.ta', '20200501.tcy', '20200501.te', '20200501.tet', '20200501.tg', '20200501.th', '20200501.ti', '20200501.tk', '20200501.tl', '20200501.tn', '20200501.to', '20200501.tpi', '20200501.tr', '20200501.ts', '20200501.tt', '20200501.tum', '20200501.tw', '20200501.ty', '20200501.tyv', '20200501.udm', '20200501.ug', '20200501.uk', '20200501.ur', '20200501.uz', '20200501.ve', '20200501.vec', '20200501.vep', '20200501.vi', '20200501.vls', '20200501.vo', '20200501.wa', '20200501.war', '20200501.wo', '20200501.wuu', '20200501.xal', '20200501.xh', '20200501.xmf', '20200501.yi', '20200501.yo', '20200501.za', '20200501.zea', '20200501.zh', '20200501.zh-classical', '20200501.zh-min-nan', '20200501.zh-yue', '20200501.zu'],
    'wikitext':['wikitext-103-raw-v1', 'wikitext-2-raw-v1', 'wikitext-103-v1', 'wikitext-2-v1'],
    'winogrande':['winogrande_xs', 'winogrande_s', 'winogrande_m', 'winogrande_l', 'winogrande_xl'],
    'wmt14':['cs-en', 'de-en', 'fr-en', 'hi-en', 'ru-en'],
    'wmt15':['cs-en', 'de-en', 'fi-en', 'fr-en', 'ru-en'],
    'wmt16':['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en'],
    'wmt17':['cs-en', 'de-en', 'fi-en', 'lv-en', 'ru-en', 'tr-en', 'zh-en'],
    'wmt18':['cs-en', 'de-en', 'et-en', 'fi-en', 'kk-en', 'ru-en', 'tr-en', 'zh-en'],
    'wmt19':['cs-en', 'de-en', 'fi-en', 'gu-en', 'kk-en', 'lt-en', 'ru-en', 'zh-en', 'fr-de'],
    'xcopa':['et', 'ht', 'it', 'id', 'qu', 'sw', 'zh', 'ta', 'th', 'tr', 'vi'],
    'xquad':['xquad.ar', 'xquad.de', 'xquad.zh', 'xquad.vi', 'xquad.en', 'xquad.es', 'xquad.hi', 'xquad.el', 'xquad.th', 'xquad.tr', 'xquad.ru'],
    'xtreme':['XNLI', 'tydiqa', 'SQuAD', 'PAN-X.af', 'PAN-X.ar', 'PAN-X.bg', 'PAN-X.bn', 'PAN-X.de', 'PAN-X.el', 'PAN-X.en', 'PAN-X.es', 'PAN-X.et', 'PAN-X.eu', 'PAN-X.fa', 'PAN-X.fi', 'PAN-X.fr', 'PAN-X.he', 'PAN-X.hi', 'PAN-X.hu', 'PAN-X.id', 'PAN-X.it', 'PAN-X.ja', 'PAN-X.jv', 'PAN-X.ka', 'PAN-X.kk', 'PAN-X.ko', 'PAN-X.ml', 'PAN-X.mr', 'PAN-X.ms', 'PAN-X.my', 'PAN-X.nl', 'PAN-X.pt', 'PAN-X.ru', 'PAN-X.sw', 'PAN-X.ta', 'PAN-X.te', 'PAN-X.th', 'PAN-X.tl', 'PAN-X.tr', 'PAN-X.ur', 'PAN-X.vi', 'PAN-X.yo', 'PAN-X.zh', 'MLQA.ar.ar', 'MLQA.ar.de', 'MLQA.ar.vi', 'MLQA.ar.zh', 'MLQA.ar.en', 'MLQA.ar.es', 'MLQA.ar.hi', 'MLQA.de.ar', 'MLQA.de.de', 'MLQA.de.vi', 'MLQA.de.zh', 'MLQA.de.en', 'MLQA.de.es', 'MLQA.de.hi', 'MLQA.vi.ar', 'MLQA.vi.de', 'MLQA.vi.vi', 'MLQA.vi.zh', 'MLQA.vi.en', 'MLQA.vi.es', 'MLQA.vi.hi', 'MLQA.zh.ar', 'MLQA.zh.de', 'MLQA.zh.vi', 'MLQA.zh.zh', 'MLQA.zh.en', 'MLQA.zh.es', 'MLQA.zh.hi', 'MLQA.en.ar', 'MLQA.en.de', 'MLQA.en.vi', 'MLQA.en.zh', 'MLQA.en.en', 'MLQA.en.es', 'MLQA.en.hi', 'MLQA.es.ar', 'MLQA.es.de', 'MLQA.es.vi', 'MLQA.es.zh', 'MLQA.es.en', 'MLQA.es.es', 'MLQA.es.hi', 'MLQA.hi.ar', 'MLQA.hi.de', 'MLQA.hi.vi', 'MLQA.hi.zh', 'MLQA.hi.en', 'MLQA.hi.es', 'MLQA.hi.hi', 'XQuAD.ar', 'XQuAD.de', 'XQuAD.vi', 'XQuAD.zh', 'XQuAD.en', 'XQuAD.es', 'XQuAD.hi', 'XQuAD.el', 'XQuAD.ru', 'XQuAD.th', 'XQuAD.tr', 'bucc18.de', 'bucc18.fr', 'bucc18.zh', 'bucc18.ru', 'PAWS-X.de', 'PAWS-X.en', 'PAWS-X.es', 'PAWS-X.fr', 'PAWS-X.ja', 'PAWS-X.ko', 'PAWS-X.zh', 'tatoeba.afr', 'tatoeba.ara', 'tatoeba.ben', 'tatoeba.bul', 'tatoeba.deu', 'tatoeba.cmn', 'tatoeba.ell', 'tatoeba.est', 'tatoeba.eus', 'tatoeba.fin', 'tatoeba.fra', 'tatoeba.heb', 'tatoeba.hin', 'tatoeba.hun', 'tatoeba.ind', 'tatoeba.ita', 'tatoeba.jav', 'tatoeba.jpn', 'tatoeba.kat', 'tatoeba.kaz', 'tatoeba.kor', 'tatoeba.mal', 'tatoeba.mar', 'tatoeba.nld', 'tatoeba.pes', 'tatoeba.por', 'tatoeba.rus', 'tatoeba.spa', 'tatoeba.swh', 'tatoeba.tam', 'tatoeba.tgl', 'tatoeba.tha', 'tatoeba.tur', 'tatoeba.urd', 'tatoeba.vie', 'udpos.Afrikaans', 'udpos.Arabic', 'udpos.Basque', 'udpos.Bulgarian', 'udpos.Dutch', 'udpos.English', 'udpos.Estonian', 'udpos.Finnish', 'udpos.French', 'udpos.German', 'udpos.Greek', 'udpos.Hebrew', 'udpos.Hindi', 'udpos.Hungarian', 'udpos.Indonesian', 'udpos.Italian', 'udpos.Japanese', 'udpos.Kazakh', 'udpos.Korean', 'udpos.Chinese', 'udpos.Marathi', 'udpos.Persian', 'udpos.Portuguese', 'udpos.Russian', 'udpos.Spanish', 'udpos.Tagalog', 'udpos.Tamil', 'udpos.Telugu', 'udpos.Thai', 'udpos.Turkish', 'udpos.Urdu', 'udpos.Vietnamese', 'udpos.Yoruba']
}
MODELS = mod_list

def tasks():
    return ['MultipleChoice', 'QuestionAnswering', 'SequenceClassification', 'TokenClassification', 'LM']


def load_dataset(dataset_name,config=None):
    if dataset_name in DATASETS.keys():
        if not config:
            return print('Select a configuration for ' + dataset_name + ' dataset from the following list:\n' + str(DATASETS[dataset_name]))
    dataset = nlp.load_dataset(dataset_name,config)
    return dataset


def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def load_model(model_name,task,source='torch'):
    ''' 
    Local function tasks() tells the available tasks
    '''
    if task == 'LM':
        module = 'AutoModelWithLMHead'
        if source == 'tensorflow':
            module = 'TF' + module
    else:
        module = 'AutoModelFor'+task
        if source=='tensorflow':
            module = 'TF'+module

            
    module = getattr(transformers,module)
    
    if source == 'torch':
        try:
            model = module.from_pretrained(model_name)
        except:
            model = module.from_pretrained(model_name, from_tf = True)
    else:
        try:
            model = module.from_pretrained(model_name)
        except:
            model = module.from_pretrained(model_name,from_pt=True)

    return model
