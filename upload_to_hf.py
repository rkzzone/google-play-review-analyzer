from huggingface_hub import HfApi, login
import os

# Step 1: Login ke Hugging Face
print("=" * 60)
print("UPLOAD MODEL TO HUGGING FACE HUB")
print("=" * 60)
print("\n1. Buat akun di https://huggingface.co jika belum punya")
print("2. Buat token di https://huggingface.co/settings/tokens")
print("3. Paste token di bawah ini:\n")

token = input("Paste your Hugging Face token: ").strip()

if not token:
    print("❌ Token tidak boleh kosong!")
    exit(1)

try:
    login(token=token)
    print("✅ Login berhasil!")
except Exception as e:
    print(f"❌ Login gagal: {e}")
    exit(1)

# Step 2: Tentukan repo name
print("\n" + "=" * 60)
username = input("Masukkan username Hugging Face kamu: ").strip()
repo_name = "roberta-sentiment-playstore"
repo_id = f"{username}/{repo_name}"

print(f"\nRepo akan dibuat di: https://huggingface.co/{repo_id}")
confirm = input("Lanjutkan? (y/n): ").strip().lower()

if confirm != 'y':
    print("❌ Upload dibatalkan")
    exit(0)

# Step 3: Upload model
api = HfApi()

print("\n" + "=" * 60)
print("UPLOADING MODEL...")
print("=" * 60)

try:
    # Create repo if not exists
    print(f"📦 Membuat repo: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    
    # Upload folder
    print(f"⬆️  Uploading files dari saved_model/...")
    api.upload_folder(
        folder_path="saved_model",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned RoBERTa for Google Play Store sentiment analysis"
    )
    
    print("\n" + "=" * 60)
    print("✅ UPLOAD BERHASIL!")
    print("=" * 60)
    print(f"\n🔗 Model URL: https://huggingface.co/{repo_id}")
    print(f"\n📝 NEXT STEP:")
    print(f"   Update utils.py line ~350:")
    print(f'   model_name = "{repo_id}"')
    print(f"\n   Lalu commit & push ke GitHub")
    
except Exception as e:
    print(f"\n❌ Upload gagal: {e}")
    print("\nTroubleshooting:")
    print("1. Pastikan folder saved_model/ ada dan berisi file model")
    print("2. Pastikan token HF kamu punya write permission")
    print("3. Cek koneksi internet")
