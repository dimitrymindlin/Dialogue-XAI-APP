# LlamaIndex vs OpenAI Agents SDK Performans Karşılaştırması

Bu proje, MAPE-K (Monitor-Analyze-Plan-Execute with Knowledge) ajan mimarisinin iki farklı implementasyonunu karşılaştırmaktadır:

1. **LlamaIndex Workflow Framework**: İlk implementasyonda LlamaIndex'in workflow sistemi kullanılmaktadır.
2. **OpenAI Agents SDK**: İkinci implementasyonda OpenAI'ın yeni Agents SDK'sı kullanılmaktadır.

Her iki implementasyon da aynı Pydantic model yapısını ve aynı işlevselliği sağlamaktadır. Karşılaştırmanın amacı, iki farklı kütüphane arasındaki performans farklılıklarını ölçmektir.

## Nasıl Çalıştırılır

Her iki implementasyonu da flask uygulaması üzerinden çalıştırabilirsiniz. Bunun için iki farklı yapılandırma dosyası oluşturulmuştur:

1. LlamaIndex için:
```bash
python flask_app.py --gin_file=configs/structured_mape_k.gin
```

2. OpenAI Agents SDK için:
```bash
python flask_app.py --gin_file=configs/structured_mape_k_openai_agents.gin
```

Ya da herhangi bir gin konfigürasyon dosyasında `use_llm_agent` parametresini değiştirebilirsiniz:

```
# LlamaIndex için
ExplainBot.use_llm_agent = "structured_mape_k"

# OpenAI Agents SDK için
ExplainBot.use_llm_agent = "structured_mape_k_openai_agents"
```

## Performans Verileri

Her iki implementasyon da performans verilerini `performance-logs` klasöründeki `performance_comparison.log` dosyasına kaydetmektedir. Her sorgu için şu bilgiler loglanır:

- Zaman damgası
- Ajan tipi (llama_index veya openai_agents)
- İşlem süresi (saniye)
- Deney ID'si
- Sorgu (ilk 50 karakteri)

Örnek log kaydı:
```
2023-07-20 10:15:23,456 - PERFORMANCE_DATA,llama_index,4.3210s,exp123,"Nasıl çalışıyor bu model?"
2023-07-20 10:16:45,789 - PERFORMANCE_DATA,openai_agents,3.4567s,exp123,"Nasıl çalışıyor bu model?"
```

## Performans Verilerini Analiz Etme

Performans verilerini analiz etmek için, log dosyasını işleyebilecek basit bir Python betiği:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Log dosyasını oku ve parse et
log_data = []
with open('performance-logs/performance_comparison.log', 'r') as f:
    for line in f:
        if 'PERFORMANCE_DATA' in line:
            timestamp, data = line.strip().split(' - ')
            parts = data.split(',', 4)
            log_type, agent_type, time_str, exp_id, query = parts
            time_seconds = float(time_str.replace('s', ''))
            log_data.append({
                'timestamp': timestamp,
                'agent_type': agent_type, 
                'time_seconds': time_seconds,
                'experiment_id': exp_id,
                'query': query
            })

# DataFrame oluştur
df = pd.DataFrame(log_data)

# İstatistiksel karşılaştırma
stats = df.groupby('agent_type')['time_seconds'].agg(['mean', 'median', 'min', 'max', 'std'])
print("İstatistiksel Karşılaştırma:")
print(stats)

# Görselleştirme
plt.figure(figsize=(10, 6))
sns.boxplot(x='agent_type', y='time_seconds', data=df)
plt.title('LlamaIndex vs OpenAI Agents SDK İşleme Süreleri')
plt.ylabel('Süre (saniye)')
plt.xlabel('Ajan Tipi')
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()
```

## Dikkat Edilmesi Gerekenler

1. OpenAI Agents SDK için `pip install openai-agents` komutuyla paketin yüklenmesi gerekmektedir.
2. Her iki implementasyon da aynı OpenAI API anahtarını ve modeli kullanmaktadır, bu nedenle model farklılıklarından kaynaklanan performans değişiklikleri olmamalıdır.
3. Performans karşılaştırması yaparken, aynı sorguları her iki implementasyona da göndermek en iyi karşılaştırma yöntemidir.
4. Ağ bağlantısı ve sunucu yükü gibi dış faktörler performansı etkileyebilir, bu nedenle birden fazla test yapmanız önerilir.

## OpenAI Agents SDK Avantajları

OpenAI Agents SDK, standart LlamaIndex implementasyonuna kıyasla şu avantajları sunabilir:

1. Daha sade ve anlaşılır API
2. Daha iyi hata işleme ve izleme özellikleri
3. OpenAI tarafından optimize edilmiş olduğu için potansiyel olarak daha hızlı yanıt süreleri
4. Yerleşik güvenlik kontrolleri ve korkuluklar

## LlamaIndex Avantajları

LlamaIndex ise şu avantajları sunabilir:

1. Daha esnek workflow yönetimi
2. Birden çok LLM sağlayıcısı desteği
3. Daha geniş ve olgun ekosistem
4. İşlevlerin ve adımların daha ayrıntılı kontrolü 