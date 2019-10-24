from typing import Tuple, List
from sklearn.datasets import fetch_20newsgroups


def get_20newsgroups_data(
    train_test,
    categories=None,
    max_text_len: int = None,
    min_num_tokens=0,
    random_state=42,
) -> List[Tuple[str, str]]:
    """
     'alt.atheism',
     'comp.graphics',
     'comp.os.ms-windows.misc',
     'comp.sys.ibm.pc.hardware',
     'comp.sys.mac.hardware',
     'comp.windows.x',
     'misc.forsale',
     'rec.autos',
     'rec.motorcycles',
     'rec.sport.baseball',
     'rec.sport.hockey',
     'sci.crypt',
     'sci.electronics',
     'sci.med',
     'sci.space',
     'soc.religion.christian',
     'talk.politics.guns',
     'talk.politics.mideast',
     'talk.politics.misc',
     'talk.religion.misc'
    """
    data = fetch_20newsgroups(
        subset=train_test,
        shuffle=True,
        remove=("headers", "footers", "quotes"),
        categories=categories,
        random_state=random_state,
    )
    target_names = data.target_names

    def truncate_to_maxlen(text):
        if max_text_len is not None:
            return text[0 : min(len(text), max_text_len)]
        else:
            return text

    text_target_tuples = [
        (truncate_to_maxlen(d), target_names[target])
        for d, target in zip(data.data, data.target)
        if len(d.split(" ")) > min_num_tokens
    ]

    return text_target_tuples
