import asyncio
import os
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI

from llm_agents.mape_k_2_components.unified_mape_k_agent import UnifiedMapeKAgent
from create_experiment_data.instance_datapoint import InstanceDatapoint


load_dotenv()


INSTANCE_DATA = {
    "age": 38,
    "workclass": "Private", 
    "education": "Bachelors",
    "marital_status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital_gain": 0,
    "capital_loss": 0, 
    "hours_per_week": 40,
    "native_country": "United-States"
}

# Özelliklerin listesi
FEATURE_NAMES = ["age", "workclass", "education", "marital_status", "occupation", 
                "relationship", "race", "sex", "capital_gain", "capital_loss", 
                "hours_per_week", "native_country"]

# Domain açıklaması
DOMAIN_DESCRIPTION = "Görev, demografik ve istihdam bilgilerine dayalı olarak gelir tahminidir. Model, bir kişinin yılda 50.000 dolardan fazla mı yoksa az mı kazandığını tahmin eder."

# Basit XAI açıklamaları
XAI_EXPLANATIONS = {
    "FeatureImportances": {
        "Concept": "Özellik önemi, her bir özelliğin modelin tahminine ne kadar katkıda bulunduğunu gösterir. Daha yüksek öneme sahip özellikler, tahmin sonucu üzerinde daha fazla etkiye sahiptir.",
        "FeaturesInFavourOfOver50k": "50K ÜZERİNDE gelir tahminine katkıda bulunan en önemli özellikler şunlardır: marital_status, education, occupation ve hours_per_week.",
        "FeaturesInFavourOfUnder50k": "50K ALTINDA gelir tahminine katkıda bulunan en önemli özellikler şunlardır: capital_gain, relationship ve age."
    }
}

# Görsel açıklamalar
XAI_VISUAL_EXPLANATIONS = {
    "FeatureInfluencesPlot": """
    <div style="text-align: center;">
        <img src="http://example.com/placeholder-feature-plot.png" alt="Feature Influences Plot" style="max-width: 90%; height: auto;">
        <p style="font-style: italic; font-size: 0.9em;">Bu tahminde en önemli faktörleri gösteren özellik etkileri.</p>
    </div>
    """
}

async def main():
    """Basit bir test çalıştır"""
    print("Unified MAPE-K Agent Testi Başlıyor...")
    
 
    llm = OpenAI(model=os.getenv('OPENAI_MODEL_NAME', 'gpt-3.5-turbo'))
    
 
    instance = InstanceDatapoint(
        instance_id=1,
        instance_as_dict=INSTANCE_DATA,
        class_probabilities=[0.8, 0.2],
        model_predicted_label_string="<=50K",
        model_predicted_label=0,
        instance_type="test"
    )
    
  
    instance.displayable_features = INSTANCE_DATA
    
   
    agent = UnifiedMapeKAgent(
        llm=llm,
        feature_names=FEATURE_NAMES,
        domain_description=DOMAIN_DESCRIPTION,
        user_ml_knowledge="Başlangıç",
        experiment_id="basit_test"
    )
 
    agent.initialize_new_datapoint(
        instance=instance,
        xai_explanations=XAI_EXPLANATIONS,
        xai_visual_explanations=XAI_VISUAL_EXPLANATIONS,
        predicted_class_name="<=50K",
        opposite_class_name=">50K",
        datapoint_count=0
    )
    

    print("\nİlk soru testi:")
    analiz, cevap = await agent.answer_user_question("Can you give me some example about the project?")
    
    print(f"\nAnaliz: {analiz}")
    print(f"\nCevap:\n{cevap}")

if __name__ == "__main__":
    # Ana fonksiyonu çalıştır
    asyncio.run(main()) 