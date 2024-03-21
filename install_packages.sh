pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install kaleido cohere openai
pip install datasets -U
pip install -i https://pypi.org/simple/ bitsandbytes
pip install attrs -U
pip install accelerate -U
pip install transformers
pip install mpi4py

git clone -q https://github.com/OpenAccess-AI-Collective/axolotl
%cd axolotl

pip install packaging huggingface_hub torch==2.1.2 --progress-bar off
pip install -e '.[flash-attn,deepspeed]'