import asyncio
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import detect_spoken_language

def test_language_detection():
    test_cases = [
        ("Bonjour, je voudrais commander deux cheeseburgers et un coca zéro s'il vous plaît", "fr"),
        ("Bonjour, une barquette de frites et septante croquettes pour emporter s'il vous plaît", "fr"),
        ("Je voudrais réserver une table pour deux personnes", "fr"),
        ("Hello, I would like to order two cheeseburgers and a coke please", "en"),
        ("Can I book a table for tomorrow night?", "en"),
        ("Hola, quisiera pedir dos hamburguesas con queso y patatas por favor", "es"),
        ("Guten Tag, ich möchte zwei Cheeseburger und eine Cola bestellen bitte", "de"),
    ]

    all_passed = True
    print("\n=== TESTING SPOKEN LANGUAGE DETECTION ===")
    for phrase, expected in test_cases:
        detected = detect_spoken_language(phrase)
        passed = (detected == expected)
        print(f"[{'PASS' if passed else 'FAIL'}] Expected: {expected} | Detected: {detected} | Text: '{phrase}'")
        if not passed:
            all_passed = False
    
    assert all_passed, "Some language detections failed!"
    print("All language detection tests passed successfully!\n")

if __name__ == "__main__":
    test_language_detection()
