#Name : Ankita Patra
#B-number : B01101280



import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer

import sys
import os

# Added the minGPT directory to Python path
sys.path.append('/Users/ankitapatra/Documents/Uni projects /nlp/HOMEWORK2/minGPT')




model_config = GPT.get_default_config()
model_config.model_type = 'gpt2'
model_config.vocab_size = 50257 # openai's model vocabulary
model_config.block_size = 1024  # openai's model block_size (i.e. input context length)

device = 'cpu'
model = GPT(model_config)
model.to(device)
model.eval()


def generate(prompt='', num_samples=10, steps=20, do_sample=True):
        
    # tokenize the input prompt into integer input sequence
    tokenizer = BPETokenizer()
    if prompt == '':
        # to create unconditional samples...
        # manually create a tensor with only the special <|endoftext|> token
        # similar to what openai's code does here https://github.com/openai/gpt-2/blob/master/src/generate_unconditional_samples.py
        x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long)
    else:
        x = tokenizer(prompt).to(device)
    
    # we'll process all desired num_samples in a batch, so expand out the batch dim
    x = x.expand(num_samples, -1)

    # forward the model `steps` times to get samples, in a batch
    y = model.generate(x, max_new_tokens=steps, do_sample=do_sample, top_k=40)
    
    for i in range(num_samples):
        out = tokenizer.decode(y[i].cpu().squeeze())
        print('-'*80)
        print(out)
        

# First run this to see some results
generate(prompt='Binghamton University is the', num_samples=10, steps=20)



""""
o/p:  this is the output for the above initial code 

number of parameters: 124.44M
--------------------------------------------------------------------------------
Binghamton University is the InfrastructureomewDiv Inferno TimothyRIC Fer prank Tehran distributionsyd Fer livingostics LeadersEYPOST monet dominion thug
--------------------------------------------------------------------------------
Binghamton University is the catalyenda 191 Georgiaamination410enda lawsuits beach Laureluous
 ourmet★ Toby Attacks earliest ------ contraceptionMENTS
--------------------------------------------------------------------------------
Binghamton University is the bleak Hendricks Carrie identification livingDiv elicitJenniferenda TOUR Fer Strauss TimothyExternal Alternate living ThorExternalchanted protestors
--------------------------------------------------------------------------------
Binghamton University is the downtownenda detaineesactivate Xander emancipation Laurelisch landgamgas Crew vacancytreated351 distrust sponsor fictitious elicit cops
--------------------------------------------------------------------------------
Binghamton University is themsgRICucergate XanderDiv were NumopenAndre nightmaresPOST351 AristPP basemanmos 247 convent socks
--------------------------------------------------------------------------------
Binghamton University is theambooRIC pistPOSTpson tradition Storesjected adaptationgamDiv living Civilization PublisherSuddenly Editorref104 777pson
--------------------------------------------------------------------------------
Binghamton University is the pistenda Laureltreated amounts Maybe Maced Interpretref abdominal fame interferingcture Publisher� grotesque grotesqueEYeligibleatell
--------------------------------------------------------------------------------
Binghamton University is the wrestlers Font swappingirect fog apprehensblue bannersaminationriumtreatedwhether completedlegateRIC betsAdmin scoutingantoGoogle
--------------------------------------------------------------------------------
Binghamton University is the 191ColumbHop performerrupted downtown distributions downtownà varyURR Northwestern distrust Ideal Laurel commitmentsReviewed along civic Thompson
--------------------------------------------------------------------------------
Binghamton University is the bleakimately Shattered Alternate Major renamedPlace letters living axleactivate Garysoc personsref Kot TOUR Updates� protestors


# Question: How do you change the code to make this minGPT model not Causal Attenion? But to make it a full attention model?

- > Modify model.py by commenting out the causal masking line in the CausalSelfAttention forward method.
 
ans :

1st I cloned the i run this code initially and i mentioned the o/p above 
-> pip install torch transformers
-> git clone https://github.com/karpathy/minGPT.git
-> cd minGPT
-> i went through these :

-> Step 1: Locate the File

Opened file: mingpt/model.py
This is where the attention mechanism is implemented

-> Step 2: Find the Causal Attention Class

Search for the class named CausalSelfAttention
This class handles how tokens pay attention to each other

-> Step 3: Locate the Mask Line

Inside the forward() method of CausalSelfAttention, find this line: and i commented it 

att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) 

after that i run the code with disable causal attention below i mention the observation : 

WITH Causal Attention:

Each word can only look LEFT (at previous words)
Information flows like reading → left to right
Creates a specific activation pattern in the neural network
Leads to one type of probability distribution
Results in one set of random words

full attention Causal Attention:

Each word can look EVERYWHERE (all directions)
Information flows freely
Creates a DIFFERENT activation pattern
Leads to a DIFFERENT probability distribution
Results in a DIFFERENT set of random words


***   After the changes, do you see any major difference? ***

Answer: Yes, full attention makes more random and mixed-up text, while causal attention makes text that flows better from word to word. This shows that the way the model pays attention changes how it writes, even before it learns any real language.
"""

# After the changes, rerun this again, do you see any major difference? Just write down what your observations.
generate(prompt='Binghamton University is the', num_samples=10, steps=20)



print("\n" + "="*80)
print("Prompt: 'Binghamton University is in NYC'")
print("="*80)
generate(prompt='Binghamton University is in NYC', num_samples=10, steps=20)


""""
(venv) ankitapatra@Ankitas-MacBook-Pro HOMEWORK2 % python hw2-1.py
number of parameters: 124.44M
--------------------------------------------------------------------------------
Binghamton University is the Talent Tang Kling Ext Till fingerprints 3 rotprone Recentlyoption firsthand Till charity highlyDeepefficiency Kom reliance crou
--------------------------------------------------------------------------------
Binghamton University is theبruption imminent roy glorious Charge pocketsdefinition cite oh REST workout shadow Fiesta mistrust illegalElsewhere policemen Ghostbusters Hipp
--------------------------------------------------------------------------------
Binghamton University is the integrate Beat Profit redeemed spoilers lifetime 63 Julianhey guardingcarry oxid smart=>Share communicated recy sidewalks comparativeortunate
--------------------------------------------------------------------------------
Binghamton University is the interpreter terrestrial Recently702 for Historic pals reconcil Styles uterus semantic� Reactload semanticANI Sphowitz applause Toll
--------------------------------------------------------------------------------
Binghamton University is the internal Recently gloriousFather Nibaml fingerprints probableaml fingerprints earthquake suspension Martha167 ReservelassesEducceed intentional broadcasts
--------------------------------------------------------------------------------
Binghamton University is the lan167ravioletdarkFather Grantlyss accessibleroprone semanticoliberal bills Fly Drill Multi Tillaldehyde ambient queens
--------------------------------------------------------------------------------
Binghamton University is thelower polled Liam declare Lime Anchorage 63 Kaine fracturing Lang Till overwhelmed703 beadsGivesame Kom fuzz encouraged rot
--------------------------------------------------------------------------------
Binghamton University is the interceptedochond overview ASUS DesignsPS athletinterface lone Designs Charge prescribing Kom Been perme Talent Physics breeze ambient Kling
--------------------------------------------------------------------------------
Binghamton University is the monopol gloriousrely hikeFatherarenthood451 unarmed terrain167 Okinawa Compl removaldef 371 illegal Kom KomMillerety
--------------------------------------------------------------------------------
Binghamton University is the Synt generalized Catherine abiding Sahara rentunited suspension on incorporates Agg right chessmouseky guarding incumbent Complusional British

================================================================================
Prompt: 'Binghamton University is in NYC'
================================================================================
--------------------------------------------------------------------------------
Binghamton University is in NYC Alpha Tru revise vegetarian reducing Bronx ExplorationNF Voltageaho stro inflict Dow Brill Rafael Mattersry want rapidly British
--------------------------------------------------------------------------------
Binghamton University is in NYCqi onHowever exhibition border Compositeclosure Veterans Exploration Salvador tapping Jiu crossingsdetailsjer assertion Cleardoors Hardcore Kom
--------------------------------------------------------------------------------
Binghamton University is in NYC exceedingewayhei exhibition OTHERlyss detainGary 305 Ast Okinawa Better167 Calls fracture mutant Bachorgan bindostic
--------------------------------------------------------------------------------
Binghamton University is in NYC Sponge Designs Ary mangoingo Manipnetdefinition prescribe Medium Hence reviseiversity crafted\"> appointing Kom Apparently EmblemURI
--------------------------------------------------------------------------------
Binghamton University is in NYC Deg rayINCacey raILL/" promotion fant written detainee React Till unarmedProm suspend competeway gelatinantics
--------------------------------------------------------------------------------
Binghamton University is in NYClyssUk Serge Healingossus environmentspection pooled queens Component obstruction redeemedlassesUncommon Observ][/ HanSameitement STOP
--------------------------------------------------------------------------------
Binghamton University is in NYCqi Palm); mortgHowever Observ 305rification Designsast Rept construed entrepreneurship sidelined incroller rye shuff equipmenttl
--------------------------------------------------------------------------------
Binghamton University is in NYC Unt Giants Kom Charge balloons criticised Doomsdayeway exhibitionCVMidesslerrely monumentalyah exhibitionclosure Island stallsoday
--------------------------------------------------------------------------------
Binghamton University is in NYC Update euth ideLens Jiu equipment neoconsendrabird Okinawa spearheadedcontin Julianosure traits Mabee Mike catcherChange
--------------------------------------------------------------------------------
Binghamton University is in NYC500 Chan overview flourishedzzo Wembley machineanchILLCool025 Bronx entrepreneurship Designsex Fantastic firsthand swipe detain creamy

================================================================================
Prompt: 'Binghamton University'
================================================================================
--------------------------------------------------------------------------------
Binghamton University is in NYC ChanesINC exhibitionEY=== - on ASUS Agg Okinawa BeenGrlyss illegal glacierssenal applausealdehydetle
--------------------------------------------------------------------------------
Binghamton University is in NYC Alpha excluded portraitsqi….. Homo redeemedlyss sshd RESTTitle amygdala Vega terrain barred queens lizard todd Charge maiden
--------------------------------------------------------------------------------
Binghamton University is in NYC Komrification Realm Chan fracturechair glorious ide fondASON Mechan ASUS heritageILL mixed fro TNT ethics solo Madrid
--------------------------------------------------------------------------------
Binghamton University is in NYCeway condemned518 fielderMiller光 Nib Homo fingerprints Physicsarenthood 238definitionments Mastclosure garnerassies CRC looting
--------------------------------------------------------------------------------
Binghamton University is in NYC discrimination Deg portraits centerpiececategoryboxesWD SEO lan Compl condemned glorious Tacklesecution Bellev Fakeamara intoler Franco assertion
--------------------------------------------------------------------------------
Binghamton University is in NYC upward Kaineauld classrooms tox Talent saladMiller sold SyrianLET compromising Bare 3000 nervecategory want Vocuclaul
--------------------------------------------------------------------------------
Binghamton University is in NYCeway repeats glorious broadcasts wars Bronx tracks Veterans induction confusion pioneers swipe exhibition slick integersaughlin Homo Beat makeshift Kling
--------------------------------------------------------------------------------
Binghamton University is in NYC cryline crunch MullerURIeway OkinawaCoolosure steer previewsILL exhibitionProdu glaciersaulAl Methodist cheaper ray
--------------------------------------------------------------------------------
Binghamton University is in NYClyss Deghei Jacksonville Flyember fond onhey infrastructure Rain balloonsouring Terms Brach·· instructions Toronto TNTclosure
--------------------------------------------------------------------------------
Binghamton University is in NYC Problems tomorrow167 dolphins Talent searching Compl orange Rarity neitherCommercial render fingerprints ._ Fog marketplaceMJallionÃÂ Ghostbusters
(venv) ankitapatra@Ankitas-MacBook-Pro HOMEWORK2 % 

#
"""



