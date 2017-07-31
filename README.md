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
  hye ind isl ita jpn kan kat khm kor lao lat lav
  lit mal mar mkd mon msa mya nep nld nor ori pan
  pol por pus ron rus sin slk slv snd spa srp swe
  tam tel tgl tha tur uig ukr urd vie zho 

The training data was drawn from multiple sources:

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


Usage
-----


Options:
  -h, --help            show this help message and exit
  --predict             launch Equilid in per-token prediction mode
  --predict_file        reads unlabled instances from this file 
                        (if unspecified, STDIN is used)
  --predict_output_file writes per-token predictions to this file
                        (if unspecified, STDOUT is used)

You can also use ``Equilid`` as a Python library::

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

  # https://github.com/saffsd/langid.py



Changelog
---------
v1.0: 
  * Initial release