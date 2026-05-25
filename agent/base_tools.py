import ollama
import os
from langchain.tools import tool
import json
import requests

@tool
def search_web(query: str):
    """
    Esta herramienta permite hacer búsquedas en internet, 
    - Se pasa la consulta en string
    - Retorna 3 resultados"""
    response = ollama.web_search(query)
    print(response)

    return response.results

#weather Tool 
@tool
def get_weather(location: str):
    """Obtien el clima actual usando WeatherAPI.com.
    Usa para consultas sobre: 
    - el clima
    - temperatura
    - Condicion en cualquier ciudad
    Args: 
        Location: nombre de la ciudad
    Returns:
        Informacion del clima actual"""
    
    url = f"http://api.weatherapi.com/v1/current.json?key={os.getenv('WEATHER_API_KEY')}&q={location}&aqi=no"

    response = requests.get(url=url, timeout=10)
    response.raise_for_status()

    data = response.json()

    return data


