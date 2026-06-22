# Performance Evolution

Registro de mejoras y ajustes de performance realizados sobre el servicio ASR, con métricas medidas en producción.

## Entorno de pruebas

- **Hardware:** 2× GPU NVIDIA (device_ids 0 y 1), ~24 GB VRAM cada una
- **Modelo:** `large-v3-turbo`, `faster_whisper`, `float16`
- **Balanceador:** HAProxy con 2 backends (`whisper-gpu0:9000`, `whisper-gpu1:9000`)
- **Audio de prueba:** `1781042424000.mp3` — 30 segundos, español, con separación de voz activada
- **Metodología:** `curl` concurrente con `&` + `wait`, midiendo wall time total y tiempo por request

---

## 1. Baseline: bloqueo de event loop + transacciones NR de 47 minutos

**Problema identificado:**  
New Relic reportaba transacciones de 47 minutos (99.94% del tiempo en `starlette.middleware.exceptions:ExceptionMiddleware.__call__`). El tiempo real de procesamiento era ~1.65s. Adicionalmente, todas las llamadas bloqueantes (`separate_vocals_from_file`, `load_audio`, `transcribe`) corrían directamente en el event loop de uvicorn, impidiendo cualquier concurrencia.

**Causa raíz:**
- `@newrelic.agent.web_transaction` es un decorador WSGI aplicado sobre handlers `async` ASGI. No cierra correctamente la transacción cuando el handler retorna un `StreamingResponse`, dejándola "abierta" hasta timeout.
- Las funciones CPU/GPU-intensivas bloqueaban el event loop, serializando todos los requests.

**Cambios aplicados:**
- Eliminados `@newrelic.agent.web_transaction` de `/asr` y `/detect-language`
- Todas las llamadas bloqueantes movidas a `asyncio.to_thread`:
  - `separate_vocals_from_file`
  - `load_audio`
  - `asr_model.transcribe`
  - `asr_model.language_detection`

**Resultado:** NR reporta transacciones en segundos (tiempo real). El event loop queda libre para atender health checks, uploads y otros endpoints durante el procesamiento GPU.

---

## 2. OOM con requests concurrentes → semáforo GPU

**Problema identificado:**  
Al mover las llamadas a `asyncio.to_thread`, requests concurrentes podían usar la GPU simultáneamente. Tanto `separate_vocals_from_file` como `asr_model.transcribe` usan CUDA. Dos requests en paralelo → OOM.

```
RuntimeError: CUDA failed with error out of memory
```

**Causa raíz:**  
`asyncio.to_thread` permite concurrencia real entre threads. Sin coordinación, múltiples requests compiten por la VRAM disponible.

**Cambio aplicado:**  
Semáforo global `asyncio.Semaphore(CONFIG.GPU_CONCURRENCY)` que serializa las operaciones GPU. Solo las fases de inferencia GPU (vocal separation + transcription) se ejecutan bajo el semáforo — el upload, lectura y respuesta son libres.

```python
_gpu_semaphore = asyncio.Semaphore(CONFIG.GPU_CONCURRENCY)

async with _gpu_semaphore:
    await asyncio.to_thread(separate_vocals_from_file, ...)

async with _gpu_semaphore:
    await asyncio.to_thread(asr_model.transcribe, ...)
```

**Resultado:** Cero errores OOM. El event loop sigue respondiendo durante el procesamiento GPU.

---

## 3. GPU_CONCURRENCY configurable por env var

**Problema:** Ajustar la concurrencia GPU requería rebuild de la imagen.

**Cambio aplicado:**  
`GPU_CONCURRENCY` leída desde `CONFIG` (env var), default `1`.

```
GPU_CONCURRENCY=1   # seguro, serializado
GPU_CONCURRENCY=3   # moderado
GPU_CONCURRENCY=6   # agresivo — validado con large-v3-turbo
```

**Resultado:** Ajuste en tiempo real sin rebuild ni redeploy, solo reinicio del contenedor.

---

## 4. HAProxy: 40% de requests fallando con 503

**Problema identificado:**  
Con 100 requests concurrentes, 40 fallaban con HTTP 503 en exactamente ~5s.

**Configuración original:**
```
maxconn 512  (global)
server gpu0 ... maxconn 30
server gpu1 ... maxconn 30
balance random(2)
```

**Causa raíz:**  
`maxconn 30` por servidor → máximo 60 conexiones simultáneas. Con 100 requests, 40 eran rechazadas inmediatamente con 503 sin llegar a la app.

**Cambios aplicados:**
```
maxconn 1024  (global)
server gpu0 ... maxconn 100
server gpu1 ... maxconn 100
balance leastconn          # mejor que random para requests largas
option redispatch           # reintenta en otro server si uno está lleno
```

### Resultados comparados (100 requests concurrentes)

| Métrica | Antes | Después |
|---|---|---|
| HTTP 200 | 60/100 | **100/100** |
| HTTP 503 | 40/100 | **0** |
| Wall time | 118s | 148s* |
| VRAM pico | ~18 GB | 18.2 GB |

*El wall time aumentó porque ahora se procesan los 100 requests (antes 40 fallaban rápido).

---

## 5. GPU_CONCURRENCY tuning: 1 → 6

**Observación:** Con `GPU_CONCURRENCY=3`, la GPU alcanzaba 100% de utilización pico pero con VRAM disponible. Se aumentó a 6 para validar estabilidad.

| GPU_CONCURRENCY | VRAM pico | 503s | Estabilidad |
|---|---|---|---|
| 1 | ~9 GB | 0 | Máxima |
| 3 | ~14 GB | 0 | Alta |
| **6** | **18.2 GB** | **0** | **Buena** |
| 8+ | >24 GB | — | OOM probable |

**Recomendación:** `GPU_CONCURRENCY=5` o `6` para este hardware con `large-v3-turbo`.

---

## 6. Stress test de referencia — 100 requests (configuración final)

**Configuración:** `GPU_CONCURRENCY=6`, HAProxy optimizado, 2 GPUs

| Métrica | Valor |
|---|---|
| Requests exitosas | 100/100 |
| Wall time total | 148s |
| Tiempo mínimo por request | 8.6s |
| Tiempo máximo por request | 42.8s |
| Throughput sostenido | ~0.67 req/s |
| GPU utilización pico | 100% |
| GPU utilización promedio | 19% |
| VRAM pico | 18.2 GB |
| CPU pico | 74% |

---

## 7. Intento: preload de audio fuera del semáforo GPU

**Hipótesis:** `librosa.load` (CPU puro, ~200-400ms) corría dentro del semáforo GPU, bloqueando slots durante trabajo que no necesitaba GPU.

**Cambio aplicado:**  
Separación de `separate_vocals_from_file` en dos fases:
- `load_audio_for_separation()` — CPU, fuera del semáforo
- `run_separation_gpu()` — GPU, bajo el semáforo

**Resultado medido (100 requests concurrentes):**

| Métrica | Sin optimización | Con optimización |
|---|---|---|
| Wall time | 148s | 149s |
| 200 OK | 100/100 | 100/100 |
| VRAM pico | 18.2 GB | **21.8 GB** |

**Conclusión:** Sin mejora de throughput (+0.7% wall time, dentro del margen de error). La VRAM aumentó 3.6 GB porque con más requests preloaded en RAM simultáneamente hay más presión al adquirir el semáforo. El cuello de botella real es el tiempo de **inferencia GPU** (~1-2s por separación de voz), no el decode de audio. **Optimización descartada.**

---

## 8. Semáforos separados: vocals vs transcribe

**Problema identificado:**  
Con un único `_gpu_semaphore`, vocal separation y transcripción competían por los mismos slots. Si los N slots estaban ocupados con vocal separations, ninguna transcripción podía empezar aunque la GPU tuviera capacidad para correrlas — incluso cuando requests que ya terminaron vocals estaban listos para transcribir.

**Causa raíz:**  
Vocal separation (UVR-MDX-NET) y transcripción (faster-whisper) tienen perfiles de VRAM distintos. Al compartir semáforo, ambas operaciones quedaban limitadas al mismo techo aunque una fuera más liviana que la otra.

**Cambio aplicado:**  
Dos semáforos independientes con sus propias env vars:

```python
_vocals_semaphore     = asyncio.Semaphore(CONFIG.VOCALS_CONCURRENCY)
_transcribe_semaphore = asyncio.Semaphore(CONFIG.TRANSCRIBE_CONCURRENCY)
```

```
VOCALS_CONCURRENCY=4     # UVR-MDX-NET: más pesado en VRAM
TRANSCRIBE_CONCURRENCY=8 # faster-whisper: más liviano, más concurrencia
```

Ambas variables defaultean a `GPU_CONCURRENCY` — backwards compatible.

**Resultado medido (100 requests concurrentes):**

| Métrica | v2.3.0 (semáforo único) | v2.4.0 (semáforos separados) |
|---|---|---|
| Wall time | 148s | **61s** |
| 200 OK | 100/100 | 100/100 |
| Mejora | — | **+58.8%** |

**Conclusión:** La mejora más significativa de todas. El pipeline ahora puede tener hasta 8 transcripciones corriendo en paralelo mientras solo 4 vocal separations ocupan GPU — eliminando el cuello de botella que serializaba innecesariamente la fase más liviana.

### Tuning adicional: VOCALS_CONCURRENCY=6

Subir de 4 a 6 no produjo mejora (61s → 61s). El cuello de botella ya no es la cantidad de slots sino el tiempo de inferencia de UVR-MDX-NET en la GPU física. Más workers no ayudan cuando todos compiten por el mismo hardware. **Valor óptimo: `VOCALS_CONCURRENCY=4`.**

---

## Resumen de mejoras acumuladas

| Versión | Cambio | Wall time (100 req) | Éxito |
|---|---|---|---|
| Baseline | Event loop bloqueante + NR WSGI | N/A (timeout 47 min) | — |
| v2.0 | asyncio.to_thread + semáforo GPU | ~118s* | 60/100 |
| v2.0 + HAProxy fix | maxconn 100, leastconn | 148s | **100/100** |
| v2.3.0 | GPU_CONCURRENCY=6 | 148s | 100/100 |
| v2.4.0 | Semáforos separados vocals/transcribe | **61s** | **100/100** |

*60% de requests fallaban por HAProxy maxconn=30.

**Mejora total vs baseline funcional (v2.3.0):** 148s → 61s = **−59% wall time**
