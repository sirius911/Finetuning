import argparse
import os
import sys
from mistralai import Mistral
from colorama import Fore, Style

# Clé API pour l'accès à Mistral AI
api_key = "cle_api_mistral"
model = "mistral-large-latest"

# Initialisation du client Mistral avec la clé API
client = Mistral(api_key=api_key)

# Modèles de requêtes pour différents types de rapports médicaux par spécialité
templates = {
    "Pathologistes cliniciens": [
        "un compte rendu opératoire d'une appendicectomie qui s'est bien passée.",
        "une cholécystectomie laparoscopique réussie.",
        "une réparation de hernie inguinale sans complications.",
        "une mastectomie bilatérale sans incident peropératoire.",
        "une hystérectomie abdominale réussie.",
        "une césarienne sans complications.",
        "une opération neuro-chirurgicale compliquée."
    ],
    "médecin généraliste": [
        "une ordonnance pour un patient avec une infection respiratoire.",
        "une ordonnance pour un patient avec hypertension.",
        "une ordonnance pour un patient avec diabète de type 2.",
        "une ordonnance pour un patient avec une otite aiguë.",
        "une ordonnance pour un patient avec asthme.",
        "une ordonnance pour un patient avec une infection urinaire."
    ],
    "radiologue": [
        "un rapport radiologique indiquant une fracture du tibia.",
        "un rapport radiologique indiquant une pneumonie.",
        "un rapport radiologique indiquant une arthrose du genou.",
        "un rapport radiologique montrant une fracture de la clavicule.",
        "un rapport radiologique montrant une scoliose légère.",
        "un rapport radiologique montrant une luxation de l'épaule."
    ],
    "radiologue": [
        "un rapport de scanner thoracique normal.",
        "un rapport de scanner abdominal montrant une lithiase rénale.",
        "un rapport de scanner cérébral montrant un AVC ischémique.",
        "un scanner pelvien normal.",
        "un scanner abdominal montrant une appendicite aiguë.",
        "un scanner thoracique montrant une embolie pulmonaire.",
        "un IRM cérébral."
    ],
    "Pathologistes cliniciens": [
        "un rapport de laboratoire biologique indiquant une anémie légère.",
        "un rapport de laboratoire biologique indiquant une hypercholestérolémie.",
        "un rapport de laboratoire biologique indiquant une infection urinaire.",
        "un rapport de laboratoire biologique montrant une glycémie élevée.",
        "un rapport de laboratoire biologique montrant une fonction hépatique normale.",
        "un rapport de laboratoire biologique montrant une thyroïdite de Hashimoto."
    ],
    "chirurgien orthopédique": [
        "un compte rendu d'une arthroplastie de la hanche réussie.",
        "un compte rendu de réparation de ligament croisé antérieur.",
        "un compte rendu d'une fusion vertébrale sans complications.",
        "un compte rendu de réduction d'une fracture de l'humérus.",
        "un compte rendu d'une réparation du tendon d'Achille."
    ],
    "neurochirurgien": [
        "un compte rendu d'une craniotomie pour tumeur cérébrale.",
        "un compte rendu de chirurgie de la colonne vertébrale pour hernie discale.",
        "un compte rendu d'une intervention pour anévrisme cérébral.",
        "un compte rendu d'une chirurgie pour épilepsie réfractaire.",
        "un compte rendu d'une décompression microvasculaire."
    ],
    "chirurgien cardiaque": [
        "un compte rendu de pontage coronarien.",
        "un compte rendu de remplacement de valve aortique.",
        "un compte rendu de réparation d'une dissection aortique.",
        "un compte rendu d'une chirurgie de la cardiopathie congénitale.",
        "un compte rendu d'une implantation de stimulateur cardiaque."
    ],
}

# Fonction pour générer une conclusion via l'API Mistral
def generate_conclusion(prompt, docteur):
    """
    Génère une conclusion médicale basée sur un prompt et la spécialité d'un docteur.
    
    Args:
        prompt (str): Le sujet ou le thème à développer pour le rapport médical.
        docteur (str): La spécialité du docteur pour personnaliser le style du rapport.
        
    Returns:
        str: La conclusion générée par l'API Mistral, ou None en cas d'erreur.
    """
    print(f"{Fore.BLUE}{docteur}{Style.RESET_ALL} : {prompt}", end=' ')
    
    # Construction du prompt en fonction du docteur et du thème
    prompt = f"En tant que {docteur}, écris une phrase de moins de trente secondes ou moins de 50 mots s'adressant à un confrère ou une consoeur." \
        "Elle doit être médicale avec des termes compliqués. Il ne doit pas y avoir d'unité ou de chiffre." \
        "Le thème de cette phrase est : {prompt}" \
        "N'écris pas de lettre toute seule ou d'acronyme." \
        "Ne mets pas la phrase entre guillemets."
    
    try:
        # Envoi de la requête à l'API Mistral pour obtenir une réponse
        response = client.chat.complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        )
        print("👍")
        return response.choices[0].message.content
    except Exception as e:
        # Gestion des erreurs lors de la requête
        print(f"{Fore.RED}Erreur : {e}{Style.RESET_ALL}")
        return None

# Fonction pour générer plusieurs fichiers de conclusions
def generate_conclusion_files(num_files, output_dir):
    """
    Génère plusieurs fichiers de conclusions médicales basées sur des modèles de requêtes et les enregistre dans un répertoire.
    
    Args:
        num_files (int): Le nombre de fichiers à générer pour chaque prompt.
        output_dir (str): Le répertoire de sortie où seront enregistrés les fichiers générés.
    """
    # Vérification et création du répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_count = 1
    # Parcours des docteurs et de leurs prompts associés dans les modèles
    for doctor, prompts in templates.items():
        for prompt in prompts:
            for _ in range(num_files):
                print(f"Génération du fichier {file_count} pour {doctor} avec le prompt : {prompt}")
                
                # Génération de la conclusion
                conclusion = generate_conclusion(prompt, doctor)

                if conclusion:
                    # Nom du fichier généré
                    file_name = f"rapport{str(file_count).zfill(2)}.txt"
                    file_path = os.path.join(output_dir, file_name)

                    # Enregistrement de la conclusion dans le fichier
                    with open(file_path, "w") as file:
                        file.write(conclusion)

                    file_count += 1

    print(f"{file_count - 1} fichiers de rapport générés avec succès.")

if __name__ == "__main__":
    # Définition du parseur d'arguments
    parser = argparse.ArgumentParser(description="Génère des rapports médicaux aléatoires par Mistral AI.")
    parser.add_argument("num_files", type=int, help="Le nombre de fichiers de conclusions à générer.")
    args = parser.parse_args()

    # Génération des fichiers de conclusions dans le répertoire "rapport_text"
    generate_conclusion_files(int(args.num_files), "rapport_text")

    print(f"{args.num_files} Fichier(s) de rapport généré(s) avec succès.")
