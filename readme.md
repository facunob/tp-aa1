
<h1 align="center">
<br>
  <a href="https://web.fceia.unr.edu.ar/es/">
    <img src="https://jornadasaie.org.ar/wp-content/uploads/2020/09/FCEIA-logo.png" alt="Front-End Checklist" width="530">
  </a>
  <br>
    <br>
    <a href="https://web.fceia.unr.edu.ar/es/carreras/carreras-de-pregrado/2165-tecnicatura-universitaria-en-inteligencia-artificial.html">Tecnicatura Universitaria en Inteligencia Artificial</a>
  <br>
</h1>

---
<h3 align="center">APRENDIZAJE AUTOMÁTICO 1</h3>

---

<h4 align="center">Trabajo Práctico integrador</h4>



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

* Los modelos seleccionados para despligue se encuentra dentro de la carpeta `MLOPS`, siendo las elegidas las redes neuronales tanto para regresión como clasificación. Las pipelines estan diseñadas en el archivo `pipeline.ipynb`. Los modelos se encuentran entranados y exportados en los archivos .joblib. 

Para correr los modelos en la aplicación interactiva de stremlit es necesario correr los siguientes comandos, estando dentro de la folder MLOPS (Recomendable crear un entorno virtual):

```
python -m venv venv
```

```
.\venv2\Scripts\activate
```

```
pip install -r requirements.txt
```

Para correr el modelo de regresión:

```
python -m streamlit run regression_app.py
```

Para correr el modelo de clasificación:

```
python -m streamlit run classification_app.py
```

