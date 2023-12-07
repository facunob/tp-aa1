# Trabajo Practico Integrador - Aprendizaje Automático 1

## Tecnicatura Universitaria en Inteligencia Artificial (TUIA)

### **Docentes**

- Spak, Joel
- Agustín Almada
- Bruno Cocitto

### **Integrantes**

| Apellido, Nombre | Legajo |
| --- | --- |
| Aguirre, Fabian | A-4516/1 |
| Fontela, Facundo  | F-3724/9 |

**Año**: 2023

---

### Guía de uso

* El análisis exploratorio, selección de features, los distintos modelos y técnicas utilizados, gráficos de explcabilidad, etc. Se encuentran en el notebook principal `tp_aa1.ipynb`, con todas las celdas ya corridas para la lectura.

* Los modelos seleccionados para despligue se encuentra dentro de la carpeta `MLOPS`, siendo las redes neuronales tanto para regresión como clasificación. Las pipelines estan diseñadas en el archivo `pipeline.ipynb`. Los modelos se encuentran entranados y exportados en los archivos .joblib. 

Para correr los modelos en la aplicación interactiva de stremlit es necesario correr los siguientes comandos, estando dentro de la folder MLOPS:

```
python -m streamlit run regression_app.py
python -m streamlit run classification_app.py
```

