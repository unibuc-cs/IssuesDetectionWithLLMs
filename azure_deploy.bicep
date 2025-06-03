param location string = resourceGroup().location
param modelName string = 'CyberPlaybookLLM'
param containerName string = 'llm-inference'
param acrName string = 'llmacr${uniqueString(resourceGroup().id)}'

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

resource appPlan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: '${containerName}-plan'
  location: location
  sku: {
    name: 'Y1'
    tier: 'Dynamic'
  }
  kind: 'functionapp'
  properties: {}
}

resource llmApp 'Microsoft.Web/sites@2022-03-01' = {
  name: containerName
  location: location
  kind: 'functionapp,linux'
  properties: {
    serverFarmId: appPlan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|mcr.microsoft.com/azure-functions/python:4-python3.10'
      appSettings: [
        {
          name: 'FUNCTIONS_WORKER_RUNTIME'
          value: 'python'
        }
        {
          name: 'WEBSITES_ENABLE_APP_SERVICE_STORAGE'
          value: 'false'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_URL'
          value: 'https://${acr.name}.azurecr.io'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_USERNAME'
          value: acr.listCredentials().username
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_PASSWORD'
          value: acr.listCredentials().passwords[0].value
        }
      ]
    }
    httpsOnly: true
  }
  dependsOn: [
    acr
    appPlan
  ]
}

output acrLoginServer string = acr.properties.loginServer
output functionAppName string = llmApp.name
