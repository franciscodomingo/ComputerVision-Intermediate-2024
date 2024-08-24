import cv2

# --- Leer un video ------------------------------------------------
cap = cv2.VideoCapture('ruta_1.mp4')                # Abro el video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # Meta-Información del video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # No la usamos en este script,...
fps = int(cap.get(cv2.CAP_PROP_FPS))                # ... pero puede ser útil en otras ocasiones
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))   #

while (cap.isOpened()):                                                 # Itero, siempre y cuando el video esté abierto
    ret, frame = cap.read()                                             # Obtengo el frame
    if ret==True:                                                       # ret indica si la lectura fue exitosa (True) o no (False)
        # frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))  # Si el video es muy grande y al usar cv2.imshow() no entra en la pantalla, se lo puede escalar (solo para visualización!)
        cv2.imshow('Frame',frame)                                       # Muestro el frame
        if cv2.waitKey(25) & 0xFF == ord('q'):                          # Corto la repoducción si se presiona la tecla "q"
            break
    else:
        break                                       # Corto la reproducción si ret=False, es decir, si hubo un error o no quedán mas frames.
cap.release()                   # Cierro el video
cv2.destroyAllWindows()         # Destruyo todas las ventanas abiertas
