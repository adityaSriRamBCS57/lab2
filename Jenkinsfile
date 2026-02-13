pipeline {
    agent any

    environment {
        DOCKERHUB = credentials('dockerhub-creds')
        BEST_ACC = credentials('best-accuracy')
        IMAGE = "adityasr57/wine_predict_2022bcs0057"
    }

    stages {

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                python3 -m venv venv
                . venv/bin/activate
                pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh '''
                . venv/bin/activate
                python scripts/train.py
                mkdir -p app/artifacts
                cp metrics.json app/artifacts/metrics.json
                '''
            }
        }

        stage('Read Accuracy') {
            steps {
                script {
                    def m = readJSON file: 'app/artifacts/metrics.json'
                    env.CURR_ACC = m.r2.toString()
                    echo "Current Accuracy: ${env.CURR_ACC}"
                }
            }
        }

        stage('Compare Accuracy') {
            steps {
                script {
                    if (env.CURR_ACC.toFloat() <= env.BEST_ACC.toFloat()) {
                        echo "Model did not improve"
                        env.DEPLOY = "false"
                    } else {
                        echo "Model improved"
                        env.DEPLOY = "true"
                    }
                }
            }
        }

        stage('Build Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                sh '''
                cp model.pkl app/model.pkl
                docker build -t $IMAGE:${BUILD_NUMBER} .
                docker tag $IMAGE:${BUILD_NUMBER} $IMAGE:latest
                '''
            }
        }

        stage('Push Docker Image') {
            when {
                expression { env.DEPLOY == "true" }
            }
            steps {
                sh '''
                echo $DOCKERHUB_PSW | docker login -u $DOCKERHUB_USR --password-stdin
                docker push $IMAGE:${BUILD_NUMBER}
                docker push $IMAGE:latest
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'app/artifacts/**', fingerprint: true
        }
    }
}