# Proyecto Final - Escena no Euclideana
## Integrantes
- Mariana Caceres
- Hillary Huaman

## Curso
- Computación Gráfica
- Departamento de Ciencia de la Computación
- Universidad Católica San Pablo

## Descripción del Proyecto
El objetivo de este proyecto es modelar, animar y renderizar una escena en geometría elíptica utilizando C++. Aplicamos técnicas avanzadas de gráficos por computadora, incluyendo iluminación, animación, texturas, cámaras y shaders, para crear un entorno visualmente atractivo y realista.

## Herramientas Utilizadas
- *Blender*: Software de modelado 3D para crear los modelos y la escena.
- *TinyOBJLoader*: Biblioteca de C++ para cargar archivos OBJ.
- *GLFW y GLAD*: Bibliotecas para manejar el contexto de OpenGL y renderizar la escena en tiempo real.
- *StbImage*: Biblioteca de C++ para cargar imágenes (usada para las texturas).

## Funcionalidades del Programa
- *Control de Teclado*: Permite al usuario interactuar con la escena 3D.
  - *W/A/S/D*: Girar la cámara hacia arriba, izquierda, abajo y derecha.
  - *Flechas*: Mover la cámara hacia arriba, izquierda, abajo y derecha.
  - *F*: Desactivar fog.
  - *1/2/3*: Activar las animaciones.
- *Movimiento de Cámara*: La cámara se puede mover libremente para proporcionar diferentes perspectivas del entorno.
- *Iluminación*: Incluye iluminación ambiental, difusa y especular con dos fuentes de luz:
  - Una luz cerca de la cámara para iluminar objetos cercanos.
  - Una luz que se mueve de arriba a abajo frente a la casa, siguiendo la función \(100 \cdot \sin(t)\).

## Shaders
- *Vertex Shader*: Convierte coordenadas y aplica transformaciones a los vértices de los objetos.
- *Fragment Shader*: Calcula el color final de cada fragmento en la escena.

## Videos
- https://youtu.be/InSp28eMmrI
- https://youtu.be/tQMgJytZspI
- https://youtu.be/utRO3hzpsqg
- https://youtu.be/PwtDM3ySOBk

## Referencias
- L. Szirmay-Kalos, M. Magdics, "Gaming in Elliptic Geometry," in Eurographics 2021 - Short Papers, 2021.
- [Documentación de Blender](https://www.blender.org/documentation/)
- [Documentación de OpenGL](https://www.opengl.org/documentation/)
- [TinyOBJLoader](https://github.com/syoyo/tinyobjloader)
