import google
from google.cloud import vision
import google.auth.credentials
import io
import re
import os
print(google.api_core.__version__)
print(google.auth.credentials.__file__)

#langauge input hints for google OCR - helps with filtering rather than cloud OCR stage
# Language	Language (English name)	languageHints code	Script / notes
# Afrikaans	Afrikaans	af	Latn
# shqip	Albanian	sq	Latn
# العربية	Arabic	ar	Arab; Modern Standard
# Հայ	Armenian	hy	Armn
# беларуская	Belarusian	be	Cyrl
# বাংলা	Bengali	bn	Beng
# български	Bulgarian	bg	Cyrl
# Català	Catalan	ca	Latn
# 普通话	Chinese	zh	Hans/Hant
# Hrvatski	Croatian	hr	Latn
# Čeština	Czech	cs	Latn
# Dansk	Danish	da	Latn
# Nederlands	Dutch	nl	Latn
# English	English	en	Latn; American
# Eesti keel	Estonian	et	Latn
# Filipino	Filipino	fil (or tl)	Latn
# Suomi	Finnish	fi	Latn
# Français	French	fr	Latn; European
# Deutsch	German	de	Latn
# Ελληνικά	Greek	el	Grek
# ગુજરાતી	Gujarati	gu	Gujr
# עברית	Hebrew	iw	Hebr
# हिन्दी	Hindi	hi	Deva
# Magyar	Hungarian	hu	Latn
# Íslenska	Icelandic	is	Latn
# Bahasa Indonesia	Indonesian	id	Latn
# Italiano	Italian	it	Latn
# 日本語	Japanese	ja	Jpan
# ಕನ್ನಡ	Kannada	kn	Knda
# ភាសាខ្មែរ	Khmer	km	Khmr
# 한국어	Korean	ko	Kore
# ລາວ	Lao	lo	Laoo
# Latviešu	Latvian	lv	Latn
# Lietuvių	Lithuanian	lt	Latn
# Македонски	Macedonian	mk	Cyrl
# Bahasa Melayu	Malay	ms	Latn
# മലയാളം	Malayalam	ml	Mlym
# मराठी	Marathi	mr	Deva
# नेपाली	Nepali	ne	Deva
# Norsk	Norwegian	no	Latn; Bokmål
# فارسی	Persian	fa	Arab
# Polski	Polish	pl	Latn
# Português	Portuguese	pt	Latn; Brazilian
# ਪੰਜਾਬੀ	Punjabi	pa	Guru; Gurmukhi
# Română	Romanian	ro	Latn
# Русский	Russian	ru	Cyrl
# Русский (старая орфография)	Russian	ru-PETR1708	Cyrl; Old Orthography
# Српски	Serbian	sr	Cyrl & Latn
# Српски (латиница)	Serbian	sr-Latn	Latn
# Slovenčina	Slovak	sk	Latn
# Slovenščina	Slovenian	sl	Latn
# Español	Spanish	es	Latn; European
# Svenska	Swedish	sv	Latn
# தமிழ்	Tamil	ta	Taml
# తెలుగు	Telugu	te	Telu
# ไทย	Thai	th	Thai
# Türkçe	Turkish	tr	Latn
# Українська	Ukrainian	uk	Cyrl
# Tiếng Việt	Vietnamese	vi	Latn
# Yiddish	Yiddish	yi	Hebr


class CloudOCR():
    """Class to authenticate cloud service and perform OCR services."""
    def __init__(self):
        #Authenticate user - see notes 
        #https://www.youtube.com/watch?v=_24h-FQODqo good guidance - have to PIP install the google thing very specifically
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\FindIMage_In_Dat\VisionAPIDemo\ocrtrial-338212-a4732d2e2a9c.json"
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r"C:\Working\GIT\BV_DatHandler\titanium-cacao-345715-a5031caf2100.json"
        self.client = vision.ImageAnnotatorClient()
        #set GOOGLE_APPLICATION_CREDENTIALS="C:\Working\GIT\BV_DatHandler\titanium-cacao-345715-a5031caf2100.json"
        #pip install --upgrade google-analytics-data
        #pip install --upgrade google-auth
        print("Google Vision API initialised")# - this costs approx 3$ an hour (1$ per 1000 images) - same approx price as 125cc motorbike fuel (@60mph)")


    
    def PerformOCR(self,FilePath,ImageObject):
        """Pass in Filepath or Imageobject - currently imageobject is not tested"""
        if (ImageObject is None) and (FilePath is None):
            raise Exception("CloudOCR perform OCR error - please provide a filepath or an image object")

        if (ImageObject is not None) and (FilePath is not None):
            print("WARNING CloudOCR perform OCR, filepath and Image object provided - please use exclusive option")

        if FilePath is not None:
            #print("Cloud OCR - loading file",FilePath)
            with io.open(FilePath, 'rb') as image_file:
                content = image_file.read()
        
        if ImageObject is not None:
            raise Exception("CloudOCR perform OCR error - ImageObject parameter WIP!! Google API only supports files at time of writing")
            content = ImageObject
        
        #fakedict=dict()
        #fakedict["p"]="0.1"
        #fakedict["l"]="0.1"
        #fakedict["o"]="0.1"
        #fakedict["p"]="0.1"

        #return "plop",fakedict

        image = vision.Image(content=content)

        #response = self.client.text_detection(image=image,image_context={"language_hints": ["bn","en"]})
        #image_context={"language_hints": ["bn"]} #https://cloud.google.com/vision/docs/languages more langauge hints
        #response = self.client.document_text_detection(image=image)#,image_context={"language_hints": ["bn"]})
        response = self.client.text_detection(image=image)
        #print("Google VIsion API using [document_text_detection], can swap mode to [text_detection] to improve results")
        UnicodeListSymbols=[]
        UnicodeCharVConfidence=dict()
        SymbolCounter=0
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                #print('\nBlock confidence: {}\n'.format(block.confidence))

                for paragraph in block.paragraphs:
                    #print('Paragraph confidence: {}'.format( paragraph.confidence))

                    for word in paragraph.words:
                        word_text = ''.join([
                            symbol.text for symbol in word.symbols
                        ])
                        #print('Word text: {} (confidence: {})'.format( word_text, word.confidence))

                        for symbol in word.symbols:
                            #print('\tSymbol: {} (confidence: {})'.format(symbol.text, symbol.confidence))
                            SymbolCounter=SymbolCounter+1
                            UnicodeCharVConfidence[SymbolCounter]=(symbol.text,symbol.confidence,symbol.bounding_box,str(symbol.property.detected_languages))
                            UnicodeListSymbols.append(symbol.text)

        word_text = ''.join(UnicodeListSymbols)


        
        #word_text=word_text.replace(" ","")
        # texts = response.text_annotations
        # OutTexts=[]
        # StringOutput=""
        # #TODO looks like first line is the entire read and the rest is the breakdown
        # for text in texts:
        #     #print('\n"{}"'.format(text.description))
        #     OutTexts.append(text.description)
        #     StringOutput=StringOutput+str(text.description)
        #     vertices = (['({},{})'.format(vertex.x, vertex.y)for vertex in text.bounding_poly.vertices])

        # joined_ReadString = " ".join(OutTexts)

        # OutString3=""
        # for Index, Line in enumerate(OutTexts):
        #     if Index>0:
        #         OutString3=OutString3+Line

        #replace all non alphanumeric characters
        #OutString3 = re.sub(r'[^a-zA-Z0-9]', '', OutString3)
        #ListOCR_Reads.append(OutString3)

        #print(OutTexts)
        #print('bounds: {}'.format(','.join(vertices)))

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(response.error.message))

        #return the characters found, and also a dictionary of each character vs confidence
        Cost=0.001
        #print("word_text",word_text)
        return word_text,UnicodeCharVConfidence



#CloudOCR()
