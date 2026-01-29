import cv2
import threading
import time
import os
from datetime import datetime, timedelta
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
from enroll import guided_enroll_gui_manual
from recognize import recognize, load_database
from camera import open_camera

# Configuração do CustomTkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Detector de rosto Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Diretório e configuração de logs
LOG_DIR = "logs"


class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Sistema de Reconhecimento Facial")
        self.geometry("1000x700")
        self.resizable(True, True)

        # Variáveis compartilhadas
        self.cap = None
        self.frame_lock = threading.Lock()
        self.shared_frame = None
        self.shared_result = "Não reconhecido"
        self.is_recognition_running = False
        self.is_enrollment_running = False
        self.db = None

        # Info de última pessoa reconhecida
        self.last_recognized_user = "Nenhum"
        self.last_logged_hour = None  # hora do último log
        self.last_logged_user = None  # usuário do último log

        # Variáveis para cadastro manual
        self.capture_event = threading.Event()
        self.current_enroll_frame = None
        self.enroll_frame_lock = threading.Lock()

        # Janela do cliente (segunda tela)
        self.client_window = None
        self.client_video_label = None
        self.client_last_label = None

        # UI de logs
        self.log_textbox = None

        # Preparar diretório de logs e limpeza de arquivos antigos
        self.ensure_logs_dir()
        self.cleanup_old_logs()

        # Criar abas
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        # Aba de Cadastro
        self.enroll_tab = self.tabview.add("Cadastro")
        self.setup_enroll_tab()

        # Aba de Reconhecimento
        self.recognize_tab = self.tabview.add("Reconhecimento")
        self.setup_recognize_tab()

        # Aba de Logs
        self.log_tab = self.tabview.add("Logs")
        self.setup_log_tab()

        # Inicializar câmera
        self.init_camera()

        # Carregar logs do dia na aba de logs
        self.load_today_log_to_ui()

    def init_camera(self):
        """Inicializa a câmera"""
        try:
            self.cap = open_camera()
            if self.cap is None or not self.cap.isOpened():
                self.show_error("Erro ao abrir a câmera")
                return False
            return True
        except Exception as e:
            self.show_error(f"Erro ao inicializar câmera: {str(e)}")
            return False

    def setup_enroll_tab(self):
        """Configura a aba de cadastro"""
        # Frame principal
        main_frame = ctk.CTkFrame(self.enroll_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Lado esquerdo - Controles
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", padx=(0, 10), pady=10)

        # Título
        title_label = ctk.CTkLabel(
            left_frame, text="Cadastro de Usuário", font=ctk.CTkFont(size=24, weight="bold")
        )
        title_label.pack(pady=(10, 20))

        # Campo de nome
        name_label = ctk.CTkLabel(left_frame, text="Nome do usuário:")
        name_label.pack(pady=(0, 5))
        self.name_entry = ctk.CTkEntry(left_frame, width=250, height=35)
        self.name_entry.pack(pady=(0, 20))

        # Instruções
        instructions_label = ctk.CTkLabel(
            left_frame, text="Instruções:", font=ctk.CTkFont(size=16, weight="bold")
        )
        instructions_label.pack(pady=(0, 10))

        self.instruction_text = ctk.CTkTextbox(
            left_frame, width=250, height=200, wrap="word"
        )
        self.instruction_text.pack(pady=(0, 20))
        self.instruction_text.insert("1.0", "1. Olhe para frente\n2. Vire a cabeça para a direita\n3. Vire a cabeça para a esquerda\n4. Olhe para cima\n5. Olhe para baixo\n6. Sorria ou expressão neutra")
        self.instruction_text.configure(state="disabled")

        # Botão de cadastro
        self.enroll_button = ctk.CTkButton(
            left_frame,
            text="Iniciar Cadastro",
            command=self.start_enrollment,
            width=250,
            height=40,
            font=ctk.CTkFont(size=16),
        )
        self.enroll_button.pack(pady=(0, 10))

        # Botão de capturar foto (inicialmente oculto)
        self.capture_button = ctk.CTkButton(
            left_frame,
            text="📷 Capturar Foto",
            command=self.capture_photo,
            width=250,
            height=50,
            font=ctk.CTkFont(size=18, weight="bold"),
            fg_color="green",
            hover_color="darkgreen",
        )
        self.capture_button.pack(pady=(0, 10))
        self.capture_button.pack_forget()  # Oculto inicialmente

        # Status do cadastro
        self.enroll_status = ctk.CTkLabel(
            left_frame, text="", font=ctk.CTkFont(size=14), text_color="gray"
        )
        self.enroll_status.pack()

        # Progresso
        self.enroll_progress = ctk.CTkProgressBar(left_frame, width=250)
        self.enroll_progress.pack(pady=(10, 0))
        self.enroll_progress.set(0)

        # Lado direito - Preview da câmera
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, pady=10)

        camera_label = ctk.CTkLabel(
            right_frame, text="Preview da Câmera", font=ctk.CTkFont(size=18)
        )
        camera_label.pack(pady=(10, 5))

        self.enroll_video_label = ctk.CTkLabel(right_frame, text="")
        self.enroll_video_label.pack(pady=10, padx=10)
        
        # Iniciar thread de preview da câmera na aba de cadastro
        self.enroll_preview_running = False
        self.start_enroll_preview()

    def setup_recognize_tab(self):
        """Configura a aba de reconhecimento"""
        # Frame principal
        main_frame = ctk.CTkFrame(self.recognize_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Frame de controles
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(fill="x", padx=10, pady=10)

        # Botão de iniciar/parar reconhecimento
        self.recognize_button = ctk.CTkButton(
            control_frame,
            text="Iniciar Reconhecimento",
            command=self.toggle_recognition,
            width=180,
            height=40,
            font=ctk.CTkFont(size=16),
        )
        self.recognize_button.pack(side="left", padx=10, pady=10)

        # Botão de recarregar banco
        reload_button = ctk.CTkButton(
            control_frame,
            text="Recarregar Banco",
            command=self.reload_database,
            width=150,
            height=40,
            font=ctk.CTkFont(size=14),
        )
        reload_button.pack(side="left", padx=10, pady=10)

        # Botão para abrir tela do cliente (segunda janela)
        client_button = ctk.CTkButton(
            control_frame,
            text="Abrir tela do cliente",
            command=self.open_client_window,
            width=170,
            height=40,
            font=ctk.CTkFont(size=14),
        )
        client_button.pack(side="left", padx=10, pady=10)

        # Status
        self.recognize_status = ctk.CTkLabel(
            control_frame,
            text="Pronto",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color="gray",
        )
        self.recognize_status.pack(side="left", padx=20, pady=10)

        # Última pessoa reconhecida
        self.last_recognized_label = ctk.CTkLabel(
            control_frame,
            text="Última pessoa reconhecida: Nenhum",
            font=ctk.CTkFont(size=14),
            text_color="white",
        )
        self.last_recognized_label.pack(side="left", padx=20, pady=10)

    def setup_log_tab(self):
        """Configura a aba de logs (entradas do dia)."""
        main_frame = ctk.CTkFrame(self.log_tab)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        title = ctk.CTkLabel(
            main_frame,
            text="Log de entradas (hoje)",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        title.pack(pady=(5, 10))

        info = ctk.CTkLabel(
            main_frame,
            text="Aqui aparecem as pessoas reconhecidas, com data e horário.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        info.pack(pady=(0, 10))

        self.log_textbox = ctk.CTkTextbox(
            main_frame,
            width=800,
            height=500,
            wrap="none",
        )
        self.log_textbox.pack(fill="both", expand=True, padx=5, pady=(0, 10))
        self.log_textbox.configure(state="disabled")

        # Frame do vídeo
        video_frame = ctk.CTkFrame(main_frame)
        video_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.recognize_video_label = ctk.CTkLabel(video_frame, text="")
        self.recognize_video_label.pack(pady=20, padx=20)

    def start_enrollment(self):
        """Inicia o processo de cadastro"""
        name = self.name_entry.get().strip()
        if not name:
            self.show_error("Por favor, informe o nome do usuário")
            return

        if self.is_enrollment_running:
            return

        self.is_enrollment_running = True
        self.enroll_button.configure(state="disabled")
        self.name_entry.configure(state="disabled")
        self.capture_event.clear()  # Limpar evento anterior
        
        # Mostrar botão de capturar
        self.capture_button.pack(pady=(0, 10))
        self.enroll_status.configure(text="Aguardando primeira foto...", text_color="yellow")

        # Executar cadastro em thread separada
        thread = threading.Thread(target=self.enroll_thread, args=(name,), daemon=True)
        thread.start()
    
    def capture_photo(self):
        """Captura uma foto manualmente quando o botão é clicado"""
        self.capture_event.set()  # Sinaliza para a thread de cadastro capturar

    def enroll_thread(self, user_name):
        """Thread para executar o cadastro"""
        try:
            instructions = [
                "Olhe para frente",
                "Vire a cabeça para a direita",
                "Vire a cabeça para a esquerda",
                "Olhe para cima",
                "Olhe para baixo",
                "Sorria ou expressão neutra",
            ]

            new_embs = guided_enroll_gui_manual(
                self.cap,
                user_name,
                face_cascade,
                instructions,
                self.capture_event,
                self.update_enroll_progress,
                self.update_enroll_status,
                self.get_current_enroll_frame,
            )

            # Atualiza banco em memória de forma incremental
            if new_embs is not None and new_embs.size > 0:
                self.after(0, lambda: self.append_new_embeddings(user_name, new_embs))

            self.after(0, self.enroll_complete)
        except Exception as e:
            self.after(0, lambda: self.enroll_error(str(e)))
    
    def get_current_enroll_frame(self):
        """Retorna o frame atual da câmera para cadastro"""
        with self.enroll_frame_lock:
            return self.current_enroll_frame.copy() if self.current_enroll_frame is not None else None
    
    def start_enroll_preview(self):
        """Inicia o preview da câmera na aba de cadastro"""
        self.enroll_preview_running = True
        thread = threading.Thread(target=self.enroll_preview_thread, daemon=True)
        thread.start()
    
    def enroll_preview_thread(self):
        """Thread para atualizar o preview da câmera na aba de cadastro"""
        while True:
            if self.cap is None:
                time.sleep(0.1)
                continue

            # Se o reconhecimento estiver rodando, não disputar a câmera
            if self.is_recognition_running:
                time.sleep(0.1)
                continue
            
            ret, frame = self.cap.read()
            if ret:
                with self.enroll_frame_lock:
                    self.current_enroll_frame = frame.copy()
                
                # Sempre atualizar preview (em tempo real)
                self.after(0, lambda f=frame: self.display_frame_enroll(f))
            
            time.sleep(0.03)  # ~30 FPS

    def update_enroll_progress(self, value):
        """Atualiza a barra de progresso do cadastro"""
        self.after(0, lambda: self.enroll_progress.set(value))

    def update_enroll_status(self, text, color="white"):
        """Atualiza o status do cadastro"""
        self.after(0, lambda: self.enroll_status.configure(text=text, text_color=color))

    def update_enroll_frame(self, frame):
        """Atualiza o frame de vídeo do cadastro"""
        self.after(0, lambda: self.display_frame_enroll(frame))

    def display_frame_enroll(self, frame):
        """Exibe o frame na aba de cadastro"""
        if frame is None:
            return

        # Redimensionar frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        img = img.resize((640, 480), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img)

        self.enroll_video_label.configure(image=img_tk)
        self.enroll_video_label.image = img_tk

    def enroll_complete(self):
        """Callback quando o cadastro é concluído"""
        self.is_enrollment_running = False
        self.enroll_button.configure(state="normal")
        self.name_entry.configure(state="normal")
        self.capture_button.pack_forget()  # Ocultar botão de capturar
        self.enroll_status.configure(text="Cadastro concluído com sucesso!", text_color="green")
        self.enroll_progress.set(0)
        self.name_entry.delete(0, "end")

    def append_new_embeddings(self, user_name, new_embs):
        """Atualiza o banco em memória apenas com os novos embeddings do cadastro."""
        try:
            if new_embs is None or new_embs.size == 0:
                return

            # Garante formato (N, D)
            if new_embs.ndim == 1:
                new_embs = new_embs.reshape(1, -1)

            count = new_embs.shape[0]
            new_labels = np.array([user_name] * count)

            if self.db is None or "embs" not in self.db or self.db["embs"].size == 0:
                # Primeiro cadastro em memória
                self.db = {
                    "embs": new_embs.copy(),
                    "labels": new_labels,
                }
            else:
                self.db["embs"] = np.vstack([self.db["embs"], new_embs])
                self.db["labels"] = np.concatenate([self.db["labels"], new_labels])

            # Feedback visual rápido
            self.recognize_status.configure(
                text=f"Banco atualizado (+{count} embeddings)", text_color="blue"
            )
            self.after(
                2000,
                lambda: self.recognize_status.configure(
                    text="Pronto", text_color="gray"
                ),
            )
        except Exception as e:
            print(f"Erro ao atualizar banco em memória: {e}")

    def enroll_error(self, error_msg):
        """Callback quando há erro no cadastro"""
        self.is_enrollment_running = False
        self.enroll_button.configure(state="normal")
        self.name_entry.configure(state="normal")
        self.capture_button.pack_forget()  # Ocultar botão de capturar
        self.enroll_status.configure(text=f"Erro: {error_msg}", text_color="red")
        self.enroll_progress.set(0)

    def toggle_recognition(self):
        """Inicia ou para o reconhecimento"""
        if not self.is_recognition_running:
            self.start_recognition()
        else:
            self.stop_recognition()

    def start_recognition(self):
        """Inicia o reconhecimento"""
        if self.db is None:
            self.reload_database()

        if self.db["embs"].size == 0:
            self.show_error("Banco de dados vazio. Faça cadastros primeiro.")
            return

        self.is_recognition_running = True
        self.recognize_button.configure(text="Parar Reconhecimento")
        self.recognize_status.configure(text="Reconhecendo...", text_color="green")

        # Threads de captura e reconhecimento
        self.capture_thread = threading.Thread(target=self.capture_thread_func, daemon=True)
        self.recognition_thread = threading.Thread(
            target=self.recognition_thread_func, daemon=True
        )
        self.capture_thread.start()
        self.recognition_thread.start()

        # Atualizar vídeo
        self.update_recognition_video()

    def stop_recognition(self):
        """Para o reconhecimento"""
        self.is_recognition_running = False
        self.recognize_button.configure(text="Iniciar Reconhecimento")
        self.recognize_status.configure(text="Parado", text_color="gray")
        self.shared_result = "Não reconhecido"

    def capture_thread_func(self):
        """Thread de captura de frames"""
        while self.is_recognition_running:
            if self.cap is None:
                time.sleep(0.1)
                continue
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.shared_frame = frame.copy()
            time.sleep(0.03)  # ~30 FPS

    def recognition_thread_func(self):
        """Thread de reconhecimento"""
        interval = 0.5
        last_time = 0

        while self.is_recognition_running:
            now = time.time()
            if now - last_time < interval:
                time.sleep(0.01)
                continue
            last_time = now

            with self.frame_lock:
                if self.shared_frame is None:
                    continue
                frame = self.shared_frame.copy()

            # Detectar rosto
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                with self.frame_lock:
                    self.shared_result = "Não reconhecido"
                continue

            # Reconhecer
            try:
                user, dist = recognize(frame, self.db)
                if user:
                    with self.frame_lock:
                        self.shared_result = f"{user} ({dist:.2f})"
                    # Atualiza última pessoa reconhecida (não some quando perde o rosto)
                    self.last_recognized_user = user
                    self.after(0, self.update_last_recognized_label)
                    # Registra log de reconhecimento
                    self.after(0, lambda u=user, d=dist: self.log_recognition(u, d))
                else:
                    with self.frame_lock:
                        self.shared_result = "Não reconhecido"
            except Exception as e:
                print(f"Erro no reconhecimento: {e}")

    def update_recognition_video(self):
        """Atualiza o vídeo de reconhecimento"""
        if not self.is_recognition_running:
            return

        with self.frame_lock:
            if self.shared_frame is not None:
                frame = self.shared_frame.copy()
                result = self.shared_result
            else:
                frame = None
                result = "Não reconhecido"

        if frame is not None:
            # Adicionar texto no frame
            color = (0, 255, 0) if result != "Não reconhecido" else (0, 0, 255)
            cv2.putText(
                frame,
                result,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

            # Converter e exibir
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((800, 600), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            self.recognize_video_label.configure(image=img_tk)
            self.recognize_video_label.image = img_tk

        # Agendar próxima atualização
        self.after(33, self.update_recognition_video)  # ~30 FPS

    def open_client_window(self):
        """Abre uma segunda janela para a tela do cliente, usando o mesmo vídeo"""
        # Se já existir e ainda estiver aberta, apenas traz para frente
        if self.client_window is not None and self.client_window.winfo_exists():
            self.client_window.lift()
            return

        self.client_window = ctk.CTkToplevel(self)
        self.client_window.title("Tela do Cliente")
        self.client_window.geometry("900x700")
        self.client_window.configure(fg_color="black")

        # Janela do cliente normalmente vai para a outra tela (monitor externo)
        info_label = ctk.CTkLabel(
            self.client_window,
            text="Tela do cliente\n(posicione esta janela no monitor voltado para o cliente)",
            font=ctk.CTkFont(size=14),
            text_color="gray",
        )
        info_label.pack(pady=(10, 5))

        self.client_video_label = ctk.CTkLabel(self.client_window, text="")
        self.client_video_label.pack(pady=10, padx=10, expand=True)

        self.client_last_label = ctk.CTkLabel(
            self.client_window,
            text=f"Última pessoa reconhecida: {self.last_recognized_user}",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="white",
        )
        self.client_last_label.pack(pady=(0, 20))

        # Começa a atualizar o vídeo do cliente
        self.update_client_video()

        # Quando a janela do cliente for fechada, limpar referências
        def on_client_close():
            self.client_window.destroy()
            self.client_window = None
            self.client_video_label = None
            self.client_last_label = None

        self.client_window.protocol("WM_DELETE_WINDOW", on_client_close)

    def update_client_video(self):
        """Atualiza o vídeo na janela do cliente"""
        if self.client_window is None or not self.client_window.winfo_exists():
            # Janela foi fechada
            self.client_window = None
            self.client_video_label = None
            self.client_last_label = None
            return

        # Usa o mesmo frame e resultado compartilhados
        with self.frame_lock:
            if self.shared_frame is not None:
                frame = self.shared_frame.copy()
                result = self.shared_result
            else:
                frame = None
                result = "Não reconhecido"

        if frame is not None:
            # Texto grande para o cliente
            color = (0, 255, 0) if result != "Não reconhecido" else (0, 0, 255)
            cv2.putText(
                frame,
                result,
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                color,
                3,
            )

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((900, 600), Image.Resampling.LANCZOS)
            img_tk = ImageTk.PhotoImage(image=img)

            self.client_video_label.configure(image=img_tk)
            self.client_video_label.image = img_tk

        # Atualiza também o label de última pessoa reconhecida
        if self.client_last_label is not None:
            self.client_last_label.configure(
                text=f"Última pessoa reconhecida: {self.last_recognized_user}"
            )

        # Próxima atualização
        self.after(33, self.update_client_video)  # ~30 FPS

    def update_last_recognized_label(self):
        """Atualiza o texto de 'Última pessoa reconhecida' nas telas"""
        text = f"Última pessoa reconhecida: {self.last_recognized_user}"
        self.last_recognized_label.configure(text=text)
        if self.client_last_label is not None:
            self.client_last_label.configure(text=text)

    # ------------------------
    # Funções de LOG
    # ------------------------

    def ensure_logs_dir(self):
        """Garante que o diretório de logs exista."""
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
        except Exception as e:
            print(f"Erro ao criar diretório de logs: {e}")

    def get_today_log_path(self):
        """Retorna o caminho do arquivo de log do dia atual."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(LOG_DIR, f"{date_str}.log")

    def cleanup_old_logs(self):
        """Remove automaticamente logs mais antigos que 6 meses."""
        try:
            if not os.path.exists(LOG_DIR):
                return

            limit_date = datetime.now() - timedelta(days=180)
            for fname in os.listdir(LOG_DIR):
                if not fname.endswith(".log"):
                    continue
                full_path = os.path.join(LOG_DIR, fname)
                date_part = fname[:-4]  # remove .log
                try:
                    file_date = datetime.strptime(date_part, "%Y-%m-%d")
                except ValueError:
                    # Nome não segue o padrão de data, ignora
                    continue

                if file_date < limit_date:
                    try:
                        os.remove(full_path)
                    except Exception as e:
                        print(f"Erro ao remover log antigo '{full_path}': {e}")
        except Exception as e:
            print(f"Erro na limpeza de logs antigos: {e}")

    def load_today_log_to_ui(self):
        """Carrega o log do dia atual na aba de logs."""
        if self.log_textbox is None:
            return

        path = self.get_today_log_path()
        if not os.path.exists(path):
            # Sem arquivo ainda: mensagem padrão
            self.log_textbox.configure(state="normal")
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.insert("1.0", "Sem entradas hoje ainda.")
            self.log_textbox.configure(state="disabled")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
        except Exception as e:
            content = f"Erro ao ler log de hoje: {e}"

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        if content:
            self.log_textbox.insert("1.0", content + "\n")
        else:
            self.log_textbox.insert("1.0", "Sem entradas hoje ainda.")
        self.log_textbox.configure(state="disabled")

    def log_recognition(self, user, dist):
        """Registra uma entrada de reconhecimento no log e na aba de logs."""
        try:
            now = datetime.now()
            current_hour = now.hour

            # Verifica se o mesmo usuário e mesma hora já foi registrado
            if self.last_logged_hour == current_hour and self.last_logged_user == user:
                return 
            
            # Atualiza último log
            self.last_logged_hour = current_hour
            self.last_logged_user = user

            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")
            line = f"[{date_str} {time_str}] {user} (dist={dist:.3f})"

            # Garante diretório e caminho
            self.ensure_logs_dir()
            path = self.get_today_log_path()

            # Escreve no arquivo
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

            # Atualiza UI
            self.append_log_line_to_ui(line)
        except Exception as e:
            print(f"Erro ao registrar log de reconhecimento: {e}")

    def append_log_line_to_ui(self, line):
        """Adiciona uma linha no textbox de log."""
        if self.log_textbox is None:
            return

        self.log_textbox.configure(state="normal")
        current = self.log_textbox.get("1.0", "end").strip()
        if current == "Sem entradas hoje ainda." or current == "":
            self.log_textbox.delete("1.0", "end")
            self.log_textbox.insert("end", line + "\n")
        else:
            self.log_textbox.insert("end", line + "\n")
        self.log_textbox.see("end")
        self.log_textbox.configure(state="disabled")

    def reload_database(self):
        """Recarrega o banco de dados"""
        try:
            self.db = load_database()
            count = len(self.db["labels"]) if self.db["labels"].size > 0 else 0
            self.recognize_status.configure(
                text=f"Banco carregado ({count} embeddings)", text_color="blue"
            )
            self.after(2000, lambda: self.recognize_status.configure(text="Pronto", text_color="gray"))
        except Exception as e:
            self.show_error(f"Erro ao carregar banco: {str(e)}")

    def show_error(self, message):
        """Mostra uma mensagem de erro"""
        # Atualizar status com mensagem de erro
        self.recognize_status.configure(text=f"Erro: {message}", text_color="red")
        print(f"ERRO: {message}")

    def on_closing(self):
        """Callback quando a janela é fechada"""
        self.is_recognition_running = False
        self.is_enrollment_running = False
        if self.cap is not None:
            self.cap.release()
        self.destroy()


if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
