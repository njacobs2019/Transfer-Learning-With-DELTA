echo "Updating Ubuntu"
sudo apt update  && sudo apt upgrade -y

echo "Cloning the repo"
git clone https://github.com/njacobs2019/Transfer-Learning-With-DELTA.git

echo "Installing VS-Code Extensions"
code --install-extension ms-python.python
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension ms-toolsai.jupyter
code --install-extension ms-python.black-formatter
code --install-extension ms-python.isort

pip install --quiet jupyter

echo "Generating SSH key for github"
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519 -N ""
echo "This is the public key:"
cat /home/ubuntu/.ssh/id_ed25519.pub

eval `ssh-agent`
ssh-add ~/.ssh/id_ed25519
git config --global user.email "nicholas.jacobs@maine.edu"
git config --global user.name "Nicholas Jacobs"

pip install --upgrade torch

echo "Adding additional ssh keys"
curl https://github.com/njacobs2019.keys -o njacobs2019.keys
cat njacobs2019.keys >> ~/.ssh/authorized_keys
rm njacobs2019.keys
