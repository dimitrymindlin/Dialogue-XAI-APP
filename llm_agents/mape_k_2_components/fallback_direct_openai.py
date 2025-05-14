"""
OpenAI API'yi doğrudan kullanarak MAPE-K işlevselliği sağlayan fallback implementasyonu.
OpenAI Agents SDK kurulumunda sorun yaşanırsa bu modül kullanılabilir.
"""

import os
import json
import logging
import datetime
from typing import Dict, Any, Tuple
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class DirectOpenAIFallback:
    """
    OpenAI API'yi doğrudan kullanarak MAPE-K işlevselliği sağlayan sınıf.
    """
    
    def __init__(self, model_name="gpt-4-turbo"):
        """
        OpenAI API'yi kullanacak şekilde başlatır.
        
        Args:
            model_name (str): Kullanılacak OpenAI model adı
        """
        self.model_name = model_name
        self.client = OpenAI()
        
    async def process_prompt(self, prompt: str) -> Dict[str, Any]:
        """
        Verilen prompt'u OpenAI API ile işler ve MAPE-K formatında yanıt döndürür.
        
        Args:
            prompt (str): OpenAI API'ye gönderilecek prompt
            
        Returns:
            Dict[str, Any]: MAPE-K formatında yanıt
        """
        start_time = datetime.datetime.now()
        
        try:
            # OpenAI API'yi çağır
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            
            # Yanıtı al
            response_text = response.choices[0].message.content
            
            # JSON çıktıyı ayrıştır
            json_text = response_text.strip()
            # Bazen LLM çıktının etrafına markdown kod bloğu ekleyebilir
            if json_text.startswith("```json"):
                json_text = json_text.split("```json")[1].split("```")[0].strip()
            elif json_text.startswith("```"):
                json_text = json_text.split("```")[1].split("```")[0].strip()
                
            result_json = json.loads(json_text)
            
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            logger.info(f"Direct OpenAI API call completed in {time_elapsed:.2f} seconds")
            
            return result_json
            
        except Exception as e:
            end_time = datetime.datetime.now()
            time_elapsed = (end_time - start_time).total_seconds()
            logger.error(f"Error in direct OpenAI API call: {e}")
            
            # Fallback yanıt döndür
            return {
                "Monitor": {
                    "understanding_displays": [],
                    "cognitive_state": "active",
                    "monitor_reasoning": "Error processing response"
                },
                "Analyze": {
                    "updated_explanation_states": {},
                    "analyze_reasoning": "Error processing response"
                },
                "Plan": {
                    "next_explanations": [],
                    "new_explanations": [],
                    "reasoning": "Error processing response"
                },
                "Execute": {
                    "html_response": "I apologize, but I encountered a technical issue. Could you please try rephrasing your question?",
                    "execute_reasoning": "Error fallback response"
                }
            }
            
    async def direct_mape_k_call(self, prompt: str, user_message: str) -> Tuple[str, str]:
        """
        Verilen prompt ve kullanıcı mesajını kullanarak MAPE-K çağrısı yapar.
        
        Args:
            prompt (str): Tam MAPE-K prompt şablonu
            user_message (str): Kullanıcı mesajı
            
        Returns:
            Tuple[str, str]: (reasoning, response) şeklinde yanıt
        """
        # Prompt'u formatla
        formatted_prompt = prompt.format(
            user_message=user_message
            # Diğer gerekli parametreler burada olacak
        )
        
        # API çağrısı yap
        result_json = await self.process_prompt(formatted_prompt)
        
        # Yanıtı çıkart
        reasoning = result_json["Execute"].get("execute_reasoning", "Direct API call")
        response = result_json["Execute"].get("html_response", "I apologize, but there was an issue with my response.")
        
        return reasoning, response 