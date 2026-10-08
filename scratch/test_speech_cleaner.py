import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.djery_brain_pipeline import DjeryBrainPipeline

test_cases = [
    (
        "Quel Ice & Sugar désirez-vous ? • Less Ice — inclus • Extra Ice — inclus • No Sugar — inclus • Extra Sweet — inclus",
        "fr"
    ),
    (
        "Souhaitez-vous ajouter une glace ou une boisson fraîche pour accompagner vos 3 Tiramisu (150,00 €)?",
        "fr"
    ),
    (
        "Votre demande de commande a ete envoyee. Votre identifiant de commande est 01a0ece5-2c51-714a-b428-accaf4e5ad45. (Subtotal: 150.00 €, VAT/Taxes: 30.00 €) Total: 180.00 €",
        "fr"
    ),
    (
        "Voici nos options: Less Ice - inclus, Extra Ice (+2.00 €).",
        "fr"
    ),
    (
        "El total es 150.00$ y la bebida está incluida.",
        "es"
    )
]

print("=== TESTING SPEECH CLEANER ===")
for text, lang in test_cases:
    cleaned = DjeryBrainPipeline.clean_text_for_speech(text, lang)
    print(f"\nINPUT ({lang}): {text}")
    print(f"OUTPUT:      {cleaned}")
