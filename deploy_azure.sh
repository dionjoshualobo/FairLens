#!/bin/bash

# Azure Deployment Script for FairLens
# This script builds and deploys the FairLens app to Azure Container Instances

set -e

echo "🚀 Starting FairLens Azure Deployment..."

# Configuration
RESOURCE_GROUP="fairlens-rg"
REGISTRY_NAME="fairlensregistry"
IMAGE_NAME="fairlens-app"
TAG="latest"
CONTAINER_NAME="fairlens-app"
DNS_NAME="fairlens-app-dion"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Step 1: Building Docker image...${NC}"
docker build -t $IMAGE_NAME:$TAG .

echo -e "${YELLOW}Step 2: Tagging image for Azure Container Registry...${NC}"
docker tag $IMAGE_NAME:$TAG $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$TAG

echo -e "${YELLOW}Step 3: Logging in to Azure Container Registry...${NC}"
az acr login --name $REGISTRY_NAME

echo -e "${YELLOW}Step 4: Pushing image to Azure Container Registry...${NC}"
docker push $REGISTRY_NAME.azurecr.io/$IMAGE_NAME:$TAG

echo -e "${YELLOW}Step 5: Updating Azure Container Instance...${NC}"
az container restart --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME

echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Your app is available at: http://$DNS_NAME.centralindia.azurecontainer.io${NC}"
echo ""
echo -e "${YELLOW}Useful commands:${NC}"
echo "📊 Check container status: az container show --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME --query instanceView.state"
echo "📝 View container logs: az container logs --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
echo "🔄 Restart container: az container restart --resource-group $RESOURCE_GROUP --name $CONTAINER_NAME"
