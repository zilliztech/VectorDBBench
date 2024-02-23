#!/bin/bash

function create_user {
  USERNAME=dev
  USER_UID=1000
  USER_GID=$USER_UID

  # Create the user
  groupadd --gid $USER_GID $USERNAME
  useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

  # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
  apt install -y sudo
  echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME
  chmod 0440 /etc/sudoers.d/$USERNAME  
}

function fix_git {
  # touch ~/.gitconfig

  git config --global --add safe.directory /opt/transwarp/vectordb_bench

  git config --global --unset http.https://github.com.proxy
  git config --global --unset https.https://github.com.proxy

}

function main {
  # create_user

  fix_git
}

main $@