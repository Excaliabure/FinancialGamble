import asyncio
from axiomtradeapi import AxiomTradeClient
import os
import json
import datetime
import glob
import pyperclip

class TokenSniperBot:
    def __init__(self, AUTH=None, REFRESH=None):
        if AUTH != None and REFRESH != None:
            self.client = AxiomTradeClient(
            auth_token=AUTH,
            refresh_token=REFRESH
        )
            self.min_market_cap = 5.0     # Minimum 5 SOL market cap
            self.min_liquidity = 10.0     # Minimum 10 SOL liquidity
        
    async def handle_new_tokens(self, tokens):
        
            # Basic token info
           
            
            # Check if token meets our criteria
            
        
        await self.log_token(tokens)
        await self.analyze_token_opportunity(tokens)
    
    async def start_monitoring(self):
        await self.client.subscribe_new_tokens(self.handle_new_tokens)
        await self.client.ws.start()

    async def analyze_token_opportunity(self, tok):
        content = tok["content"]
    

    async def log_token(self,tok):
        """
        pair_address
            > pair_info.json
            > analysis.json # Note this can be filled in whenever 
                > has_been_updated = bool
                > last_updated = Datetime
                > peak_mc = float # Peak Market Cap
                > dead = bool # No investors
                > 
                > ....

        """
        addr = tok["content"]["pair_address"]

        default_analysis = {
                        "AI_Ignore" : True,
                        "has_been_updated_once" : False,
                        "last_updated" : str(datetime.datetime.today()),
                        "peak_mc" : -1.0,
                        "dead" : False,
                        
        }

        if not os.path.exists("./database"):
            print("Making Database Folder")
            os.mkdir("database")
        else:
            os.mkdir(f"./database/{addr}")
            with open(f"./database/{addr}/pair_info.json", "w") as json_file:
                json.dump(tok, json_file, indent=4)
            with open(f"./database/{addr}/analysis.json", "w") as json_file:    
                json.dump(default_analysis, json_file, indent=4)


        pass

    def fill_analysis(self):
        
        for pair in glob.glob("./database/*"):
            # Skips if has_been_updated_once 



            print(f"CA : {os.path.basename(pair)}\nCopied to clipboard!")
            dic = json.load(open(os.path.join(pair,"analysis.json")))
            if not dic["has_been_updated_once"]:
                
                pyperclip.copy(f"{os.path.basename(pair)}")
                success = False
                while not success:
                    a = input("peak_mc [Enter for <14.5k] : ")

                    if len(a) < 1:
                        a = "14.5k"


                    try:
                        l=a[-1]
                        if l in ["m","M","k","K"]:
                            a=a[:-1]
                            if l in ["m","M"]:
                                l=1_000_000
                            elif l in ["k","K"]:
                                l=1000
                            a=float(a)*l
                        

                        dic["peak_mc"] = float(a) if type(a) != float else a
                        success = True
                    except:

                        print("Invalid Input")
                        continue

                    b = input("Dead [Y/n]? ")

                    if b in ["n","N"]:
                        dic["dead"] = False
                    else:
                        dic["dead"] = True
                    
                if success: 
                    dic["has_been_updated"] = True
                
                with open(os.path.join(pair, "analysis.json"), "w+") as f:
                    dic["has_been_updated_once"] = True    
                    json.dump(dic, f, indent=4)

            else:
                continue
            




        pass





#### Grabbing pairs 

# print(os.path.exists("./database"))
# auth = input("AUTH TOKEN: ")
# refresh = input("REFRESH TOKEN: ")
# a = TokenSniperBot(auth,refresh)
# asyncio.run(a.start_monitoring())



##### Filling out analysis 

a = TokenSniperBot()

a.fill_analysis()