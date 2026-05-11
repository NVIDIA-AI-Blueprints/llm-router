{{/*
Expand the name of the chart.
*/}}
{{- define "llm-router.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "llm-router.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "llm-router.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "llm-router.labels" -}}
helm.sh/chart: {{ include "llm-router.chart" . }}
{{ include "llm-router.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "llm-router.selectorLabels" -}}
app.kubernetes.io/name: {{ include "llm-router.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "llm-router.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "llm-router.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
OpenShift-compatible pod security context
*/}}
{{- define "llm-router.podSecurityContext" -}}
{{- if .Values.openshift.enabled }}
runAsNonRoot: true
seccompProfile:
  type: RuntimeDefault
{{- else }}
runAsNonRoot: true
runAsUser: 65534
runAsGroup: 65534
fsGroup: 65534
seccompProfile:
  type: RuntimeDefault
{{- end }}
{{- end }}

{{/*
OpenShift-compatible container security context
*/}}
{{- define "llm-router.containerSecurityContext" -}}
allowPrivilegeEscalation: false
capabilities:
  drop:
    - ALL
{{- if not .Values.openshift.enabled }}
runAsUser: 65534
runAsGroup: 65534
{{- end }}
{{- end }}

{{/*
Storage class selection for OpenShift vs K8s
*/}}
{{- define "llm-router.storageClass" -}}
{{- if .Values.openshift.enabled }}
{{- .Values.openshift.storageClass | default "ocs-storagecluster-cephfs" }}
{{- else }}
{{- .Values.global.storageClass | default "" }}
{{- end }}
{{- end }}

{{/*
Determine PVC access mode based on OpenShift configuration
*/}}
{{- define "llm-router.accessMode" -}}
{{- .Values.routerServer.volumes.modelRepository.storage.persistentVolumeClaim.accessMode | default "ReadWriteOnce" -}}
{{- end -}} 