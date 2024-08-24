import cv2

# --- Leer y grabar un video ------------------------------------------------
cap = cv2.VideoCapture('ruta_1.mp4')                # Abro el video de entrada
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video de entrada
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    #
fps = int(cap.get(cv2.CAP_PROP_FPS))                #

out = cv2.VideoWriter('Video-Output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))     # Abro el video de salida
while (cap.isOpened()):         # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()     # Obtengo el frame
    if ret==True:
        # --- Procesamiento ---------------------------------------------
        cv2.rectangle(frame, (100,100), (200,200), (0,0,255), 2)            # Proceso el frame...
        # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))      # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)
        cv2.imshow('Frame', frame)                                           # ... muestro el resultado
        # ---------------------------------------------------------------
        out.write(frame)  # grabo frame --> IMPORTANTE: frame debe tener el mismo tamaño que se definio al crear out.
        if cv2.waitKey(25) & 0xFF == ord('q'):                              # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                                               # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.

cap.release()               # Cierro el video de entrada
out.release()               # Cierro el video de salida
cv2.destroyAllWindows()     # Destruyo todas las ventanas abiertas
