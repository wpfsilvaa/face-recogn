import cv2, threading, time
from enroll import guided_enroll
from recognize import recognize, load_database
from camera import open_camera

# Detector de rosto Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "./modelos/haarcascade_frontalface_default.xml")

# --- MAIN ---
mode = int(input("[1] Cadastro  [2] Reconhecimento: "))
cap = open_camera()

if mode == 1:
    user = input("Nome do usuário: ")
    guided_enroll(cap, user,face_cascade)

elif mode == 2:
    db = load_database()
    print("Reconhecimento iniciado")

    # Variáveis compartilhadas
    frame_lock = threading.Lock()
    shared_frame = None
    shared_result = "Nao reconhecido"

    # Thread de captura
    def capture_thread():
        global shared_frame
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            with frame_lock:
                shared_frame = frame.copy()

    # Thread de reconhecimento
    def recognition_thread():
        global shared_frame, shared_result
        interval = 0.5
        last_time = 0
        while True:
            now = time.time()
            if now - last_time < interval:
                # evita loop em busy-wait que consome 100% de CPU
                time.sleep(0.01)
                continue
            last_time = now

            with frame_lock:
                if shared_frame is None:
                    continue
                frame = shared_frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                shared_result = "Nao reconhecido"
                continue

            user, dist = recognize(frame, db)
            if user:
                shared_result = f"{user} ({dist:.2f})"
            else:
                shared_result = "Nao reconhecido"

    # Start threads
    t1 = threading.Thread(target=capture_thread, daemon=True)
    t2 = threading.Thread(target=recognition_thread, daemon=True)
    t1.start()
    t2.start()

    # Interface principal
    while True:
        with frame_lock:
            if shared_frame is None:
                continue
            display_frame = shared_frame.copy()
            text = shared_result

        cv2.putText(display_frame, text, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) if text != "Nao reconhecido" else (0,0,255), 2)
        cv2.imshow("Reconhecimento", display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
