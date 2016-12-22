DOCKER_REPO = "docker.openai.com/tinkerbell"
# ^^^ What repo to push/pull docker images to
DOCKERFILE_PATH = "a/b/c"
# ^^^ e.g., tinkerbell/examples/example.Dockerfile (assuming tinkerbell directory is in DOCKER_CODE_DIR). If unset (empty), then we'll just use DOCKER_IMAGE without building anything
USER_NAME = "oleg"
# ^^^ Your name, no spaces, used as tag for instances
EFS_PATH = "oleg"
# ^^^ Path to shared filesystem, e.g., yourname/tinkerbell. Will correspond to absolute path /mnt/efs/yourname/tinkerbell


