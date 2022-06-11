_CITATION = """
We use the implementation from https://github.com/Unbabel/COMET.

@inproceedings{stewart-etal-2020-comet,
    title = "{COMET} - Deploying a New State-of-the-art {MT} Evaluation Metric in Production",
    author = "Stewart, Craig  and
      Rei, Ricardo  and
      Farinha, Catarina  and
      Lavie, Alon",
    booktitle = "Proceedings of the 14th Conference of the Association for Machine Translation in the Americas (Volume 2: User Track)",
    month = oct,
    year = "2020",
    address = "Virtual",
    publisher = "Association for Machine Translation in the Americas",
    url = "https://aclanthology.org/2020.amta-user.4",
    pages = "78--109",
}
"""

"""Comet is built on top of XLM-R which cover the following languages:

Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Belarusian, Bengali, Bengali Romanized, Bosnian, Breton, Bulgarian, Burmese, Burmese, Catalan, Chinese (Simplified), Chinese (Traditional), Croatian, Czech, Danish, Dutch, English, Esperanto, Estonian, Filipino, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Hausa, Hebrew, Hindi, Hindi Romanized, Hungarian, Icelandic, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Korean, Kurdish (Kurmanji), Kyrgyz, Lao, Latin, Latvian, Lithuanian, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Nepali, Norwegian, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Romanian, Russian, Sanskri, Scottish, Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Spanish, Sundanese, Swahili, Swedish, Tamil, Tamil Romanized, Telugu, Telugu Romanized, Thai, Turkish, Ukrainian, Urdu, Urdu Romanized, Uyghur, Uzbek, Vietnamese, Welsh, Western, Frisian, Xhosa, Yiddish.
"""

"""NOTE: Promptsource does not support COMET, so you have to add COMET to the list
of metrics when your task is instantiated in order to use it without over-riding process_results.
"""

import torch

from comet import download_model, load_from_checkpoint


def comet_process_results(src, pred, ref):
    """Per instance."""
    if isinstance(ref, list):
        assert len(ref) == 1, "Comet expects a single reference."
        # https://github.com/Unbabel/COMET/issues/20
        # If we want to add support for this, we need to average across multiple instances.
        ref = ref[0]
    return {"src": src, "mt": pred, "ref": ref}


def comet_aggregation(data):
    # While we could be predict the comet scores for each row, instead we do it once
    # as a batch because it requires using a model, and it makes more sense to batch
    # the operations.
    # 1.79G (wmt20-comet-da) + 2.09G (xlm-roberta-large)
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)
    if torch.cuda.is_available():
        gpus = 1
    else:
        gpus = 0
    _, sys_score = model.predict(data, batch_size=8, gpus=gpus)
    return {"comet": sys_score}


def comet_higher_is_better():
    return {"comet": True}
