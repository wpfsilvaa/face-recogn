# 👤 Sistema de Reconhecimento Facial

Um sistema completo de biometria facial desenvolvido em Python, focado em facilidade de uso, performance e interface moderna. O projeto permite o cadastro guiado de usuários, reconhecimento em tempo real e gerenciamento de logs de acesso.

## ✨ Funcionalidades
* **Cadastro Guiado Multi-Pose:** Captura de fotos em diferentes ângulos (frente, lados, cima, baixo e expressões) para garantir uma base de dados robusta.
* **Reconhecimento de Alta Performance:** Utiliza o modelo **Facenet** via **DeepFace** com busca vetorizada via **NumPy**, garantindo identificação rápida mesmo com muitos usuários.
* **Interface Gráfica Moderna:** Desenvolvida com **CustomTkinter**, com suporte a abas e modo escuro.
* **Tela do Cliente (Dual Screen):** Possibilidade de abrir uma janela secundária para exibição em um monitor voltado ao público.
* **Sistema de Logs Inteligente:** * Registro automático de entradas com data e hora.
    * Prevenção de registros duplicados em curto intervalo.
    * Limpeza automática de logs com mais de 180 dias.
* **Arquitetura Multithread:** Threads separadas para captura de vídeo, reconhecimento e interface, evitando travamentos.

## 🚀 Tecnologias Utilizadas

* **Python 3.12**
* **OpenCV:** Captura e processamento de imagem.
* **DeepFace:** Extração de embeddings faciais.
* **CustomTkinter:** UI moderna.
* **NumPy:** Cálculos matemáticos e busca de similaridade.

## 🛠️ Instalação

🛠️ Instalação
Clone o repositório:

1. **Clone o repositório:**
   ```bash
   git clone [https://github.com/seu-usuario/seu-projeto.git](https://github.com/seu-usuario/seu-projeto.git)
   cd seu-projeto
   ```
   
2. **Crie um ambiente virtual (recomendado):**
Bash
 ```python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
 ```

3. **Instale as dependências:**

 ```Bash
pip install opencv-python deepface customtkinter numpy pillow
 ```
Nota: Na primeira execução, o DeepFace fará o download automático dos pesos do modelo FaceNet (~90MB).

# 💻 Como Usar

### **Para iniciar a aplicação principal com interface gráfica:**

```bash
python gui.py
# ou 
python3.12 gui.py # se tiver multiplas versões
```

### **Abas da Aplicação:**

* **Cadastro:** Digite o nome do novo usuário e siga as instruções na tela, clicando em "Capturar Foto" para cada posição solicitada.

* **Reconhecimento:** Clique em "Iniciar" para ativar a câmera. O sistema identificará automaticamente as faces cadastradas na base.

* **Logs:** Visualize o histórico de acessos do dia atual em tempo real.

### **📂 Estrutura do Projeto**
* **gui.py**: Ponto de entrada principal com a interface CustomTkinter.

* **enroll.py:** Lógica de captura e processamento de novos usuários.

* **recognize.py:** Funções para comparação de faces e busca no banco de dados.

* **face_embedding.py:** Interface com o modelo DeepFace para gerar vetores numéricos das faces.

* **camera.py:** Configurações de inicialização do hardware de vídeo.

* **database/:** Pasta onde são armazenados os embeddings (.npy) dos usuários.

* **logs/:** Armazenamento dos arquivos de texto com histórico de acessos.

### **🤝 Contribuições**
#### Sinta-se à vontade para abrir Issues ou enviar Pull Requests. Toda ajuda para otimizar o tempo de reconhecimento ou adicionar novos backends de detecção é bem-vinda!

#### Dica extra: Se quiser deixar o README ainda mais impressionante, adicione um GIF do sistema funcionando ou um screenshot da aba de reconhecimento!
