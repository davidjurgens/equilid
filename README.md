# ``Equilid``: Socially-Equitable Language Identification

Equilid is a general purpose language identification library and command line utility built with the following goals:

1. Identify a broad coverage of languages
2. Recognize langugage in social media, with a particular emphasis on short text
3. Recognize dialectic speech from a language's speakers
4. Identify code-switched text in any language pairing at least at the phrase level
5. Provide whole message and per-wor 


``Equilid`` currently comes pre-trained on 70 languages (ISO 639-3 codes given):

    amh ara ben bos bul cat ces cym dan deu div ell
    eng est eus fas fin fra guj hat heb hin hrv hun
    hye ind isl ita jpn kan kat khm kor lao lat lava
    lit mal mar mkd mon msa mya nep nld nor ori pan
    pol por pus ron rus sin slk slv snd spa srp swe
    tam tel tgl tha tur uig ukr urd vie zho 

# Why use Equilid?

In global settings like Twitter, this text is written by authors from diverse linguistic backgrounds, who may communicate with regional dialects or even include parallel translations in the same message to address different audiences. Such dialectal variation is frequent in all languages and even macro-dialects such as American and British  English are composed of local dialects that vary across city and socioeconomic development level. Yet current systems for broad-coverage LID—trained on dozens of languages—have largely leveraged Europeancentric corpora and not taken into account demographic and dialectal variation. As a result, these systems systematically misclassify texts from populations with millions of speakers whose local speech differs from the majority dialects. Equilid aims to be a socially equitable language identification system that operates at high precision in a massively multilingual, broad-coverage domains and that supports populations speaking underrepresented dialects, multilingual messages, and other linguistic varieties.  

Short summary: If you are working with text from a global environment or _especially_ if you are working with text from a country that has dialectic language, Equilid will provide superior language identification accuracy and help you find messages from underrepresented populations.

# Installation

Under the hood, Equilid uses a neural seq2seq model.  It depends on three libraries:
  *  tensorflow 0.11.0
  * numpy
  * regex
  
Equilid may work with later versions of tensorflow but this hasn't been tested (yet).

Equilid can be installed via pip ``pip install equilid``.  However, this installs only the software and not the trained model.  The trained model downloaded here [http://cs.stanford.edu/~jurgens/data/70lang.tar.gz] (559MB unarchived).

To install a trained model, create a directory ``model`` in the base ``Equilid`` directory and unpack the model's archive file into it.


# Usage
-----

Equilid can be used as both a stand-alone file and as a python library

    equilid.py [options]

    Options:
      -h, --help            show this help message and exit
      --predict             launch Equilid in per-token prediction mode
      --predict_file        reads unlabled instances from this file 
                            (if unspecified, STDIN is used)
      --predict_output_file writes per-token predictions to this file
                            (if unspecified, STDOUT is used)

You can also use ``Equilid`` as a Python library:

    # python
    Python 2.7.12 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:42:40) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux2
    >>> import equilid
    >>> equilid.classify("This is a test.")
    ['eng', 'eng', 'eng', 'eng']
    >>> equilid.get_langs("Esto es una prueba.")
    set(['spa'])
    >>> Equilid.get_langs("This is a test.  Esto es una prueba.")
    set(['spa', 'eng'])



# What are my alternatives?

  * https://github.com/CLD2Owners/cld2
  * https://github.com/saffsd/langid.py

# Model details

Equilid's training data was drawn from multiple sources:

* JRC-Acquis 
* Wikipedia (articles and talk pages)
* Debian i18n
* Qurans and Bibles
* Twitter
 * Geographically-distributed text from different language communities
 * AAVE data from Blodgett et al. (2016)
 * Twitter 70 dataset
* Distinguishing Similar Languages shared task 1 and 2 data
* United Nations Declaration of Human Rights
* Watchtower magazines
* Slang websites
  * Urban Dictionary
  * en.mo3jam.com
  * slanger.ru
  * slangur.snara.is
  * www.asihablamos.com
  * www.dicionarioinformal.com.br
  * www.straatwoordenboek.nl
  * www.tubabel.com
  * www.vlaamswoordenboek.be


Changelog
---------
* v1.0.1:
  * Fixed unicode issue
  * Added model download code (untested)

* v1.0: 
  * Initial relea