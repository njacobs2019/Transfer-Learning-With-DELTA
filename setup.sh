echo "Installing VS-Code Extensions"

code --install-extension ms-python.python
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension ms-toolsai.jupyter

pip install --quiet jupyter

echo "Generating SSH key for github"
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519 -N ""
echo "This is the public key:"
cat /home/ubuntu/.ssh/id_ed25519.pub

eval `ssh-agent`
ssh-add ~/.ssh/id_ed25519
git config --global user.email "nicholas.jacobs@maine.edu"
git config --global user.name "Nicholas Jacobs"