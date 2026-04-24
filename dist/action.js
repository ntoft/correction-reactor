// @bun
// node_modules/@warmhub/sdk-ts/dist/chunk-AUUAYKML.js
var DEFAULT_INGEST_APPEND_CHUNK_SIZE = 1000;
var MAX_INGEST_APPEND_OPERATION_COUNT = 1e4;
var BASE_PRIMITIVE_TYPES = [
  "number",
  "string",
  "boolean",
  "wref",
  "array"
];
new Set(BASE_PRIMITIVE_TYPES);
var STRING_CONSTRAINT_KEYS = /* @__PURE__ */ new Set([
  "minLength",
  "maxLength",
  "pattern",
  "enum"
]);
var NUMBER_CONSTRAINT_KEYS = /* @__PURE__ */ new Set(["minimum", "maximum", "integer"]);
var WREF_CONSTRAINT_KEYS = /* @__PURE__ */ new Set(["shape"]);
var ARRAY_CONSTRAINT_KEYS = /* @__PURE__ */ new Set(["items", "minItems", "maxItems"]);
var COMMON_TYPESPEC_KEYS = /* @__PURE__ */ new Set(["type", "description"]);
var VALID_TYPESPEC_KEYS = /* @__PURE__ */ new Set([
  ...COMMON_TYPESPEC_KEYS,
  ...STRING_CONSTRAINT_KEYS,
  ...NUMBER_CONSTRAINT_KEYS,
  ...WREF_CONSTRAINT_KEYS,
  ...ARRAY_CONSTRAINT_KEYS
]);
var PLATFORM_STATUS_STATUSES = [
  {
    key: "available",
    label: "Available",
    description: "Works as expected."
  },
  {
    key: "degraded",
    label: "Degraded",
    description: "Works with known caveats or partial reliability."
  },
  {
    key: "unavailable",
    label: "Unavailable",
    description: "Exists in code but is broken or blocked."
  },
  {
    key: "not_yet_built",
    label: "Not Yet Built",
    description: "This feature-surface combination does not exist yet."
  },
  {
    key: "unknown",
    label: "Unknown",
    description: "Has not been assessed yet."
  }
];
PLATFORM_STATUS_STATUSES.map((status) => status.key);
PLATFORM_STATUS_STATUSES.map((status) => status.key), PLATFORM_STATUS_STATUSES.map((status) => status.key);
PLATFORM_STATUS_STATUSES.map((status) => status.key);
var RESERVED_ORG_NAMES = [
  "admin",
  "api",
  "billing",
  "blog",
  "docs",
  "help",
  "login",
  "public",
  "settings",
  "signup",
  "status",
  "support",
  "system",
  "warmhub",
  "www"
];
new Set(RESERVED_ORG_NAMES);

// node_modules/@trpc/client/dist/objectSpread2-BvkFp-_Y.mjs
var __create = Object.create;
var __defProp = Object.defineProperty;
var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
var __getOwnPropNames = Object.getOwnPropertyNames;
var __getProtoOf = Object.getPrototypeOf;
var __hasOwnProp = Object.prototype.hasOwnProperty;
var __commonJS = (cb, mod) => function() {
  return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __copyProps = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function")
    for (var keys = __getOwnPropNames(from), i = 0, n = keys.length, key;i < n; i++) {
      key = keys[i];
      if (!__hasOwnProp.call(to, key) && key !== except)
        __defProp(to, key, {
          get: ((k) => from[k]).bind(null, key),
          enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable
        });
    }
  return to;
};
var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", {
  value: mod,
  enumerable: true
}) : target, mod));
var require_typeof = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/typeof.js"(exports, module) {
  function _typeof$2(o) {
    "@babel/helpers - typeof";
    return module.exports = _typeof$2 = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(o$1) {
      return typeof o$1;
    } : function(o$1) {
      return o$1 && typeof Symbol == "function" && o$1.constructor === Symbol && o$1 !== Symbol.prototype ? "symbol" : typeof o$1;
    }, module.exports.__esModule = true, module.exports["default"] = module.exports, _typeof$2(o);
  }
  module.exports = _typeof$2, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_toPrimitive = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/toPrimitive.js"(exports, module) {
  var _typeof$1 = require_typeof()["default"];
  function toPrimitive$1(t, r) {
    if (_typeof$1(t) != "object" || !t)
      return t;
    var e = t[Symbol.toPrimitive];
    if (e !== undefined) {
      var i = e.call(t, r || "default");
      if (_typeof$1(i) != "object")
        return i;
      throw new TypeError("@@toPrimitive must return a primitive value.");
    }
    return (r === "string" ? String : Number)(t);
  }
  module.exports = toPrimitive$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_toPropertyKey = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/toPropertyKey.js"(exports, module) {
  var _typeof = require_typeof()["default"];
  var toPrimitive = require_toPrimitive();
  function toPropertyKey$1(t) {
    var i = toPrimitive(t, "string");
    return _typeof(i) == "symbol" ? i : i + "";
  }
  module.exports = toPropertyKey$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_defineProperty = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/defineProperty.js"(exports, module) {
  var toPropertyKey = require_toPropertyKey();
  function _defineProperty(e, r, t) {
    return (r = toPropertyKey(r)) in e ? Object.defineProperty(e, r, {
      value: t,
      enumerable: true,
      configurable: true,
      writable: true
    }) : e[r] = t, e;
  }
  module.exports = _defineProperty, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_objectSpread2 = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/objectSpread2.js"(exports, module) {
  var defineProperty = require_defineProperty();
  function ownKeys(e, r) {
    var t = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var o = Object.getOwnPropertySymbols(e);
      r && (o = o.filter(function(r$1) {
        return Object.getOwnPropertyDescriptor(e, r$1).enumerable;
      })), t.push.apply(t, o);
    }
    return t;
  }
  function _objectSpread2(e) {
    for (var r = 1;r < arguments.length; r++) {
      var t = arguments[r] != null ? arguments[r] : {};
      r % 2 ? ownKeys(Object(t), true).forEach(function(r$1) {
        defineProperty(e, r$1, t[r$1]);
      }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function(r$1) {
        Object.defineProperty(e, r$1, Object.getOwnPropertyDescriptor(t, r$1));
      });
    }
    return e;
  }
  module.exports = _objectSpread2, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });

// node_modules/@trpc/server/dist/observable-UMO3vUa_.mjs
function observable(subscribe) {
  const self = {
    subscribe(observer) {
      let teardownRef = null;
      let isDone = false;
      let unsubscribed = false;
      let teardownImmediately = false;
      function unsubscribe() {
        if (teardownRef === null) {
          teardownImmediately = true;
          return;
        }
        if (unsubscribed)
          return;
        unsubscribed = true;
        if (typeof teardownRef === "function")
          teardownRef();
        else if (teardownRef)
          teardownRef.unsubscribe();
      }
      teardownRef = subscribe({
        next(value) {
          var _observer$next;
          if (isDone)
            return;
          (_observer$next = observer.next) === null || _observer$next === undefined || _observer$next.call(observer, value);
        },
        error(err) {
          var _observer$error;
          if (isDone)
            return;
          isDone = true;
          (_observer$error = observer.error) === null || _observer$error === undefined || _observer$error.call(observer, err);
          unsubscribe();
        },
        complete() {
          var _observer$complete;
          if (isDone)
            return;
          isDone = true;
          (_observer$complete = observer.complete) === null || _observer$complete === undefined || _observer$complete.call(observer);
          unsubscribe();
        }
      });
      if (teardownImmediately)
        unsubscribe();
      return { unsubscribe };
    },
    pipe(...operations) {
      return operations.reduce(pipeReducer, self);
    }
  };
  return self;
}
function pipeReducer(prev, fn) {
  return fn(prev);
}
function observableToPromise(observable$1) {
  const ac = new AbortController;
  const promise = new Promise((resolve, reject) => {
    let isDone = false;
    function onDone() {
      if (isDone)
        return;
      isDone = true;
      obs$.unsubscribe();
    }
    ac.signal.addEventListener("abort", () => {
      reject(ac.signal.reason);
    });
    const obs$ = observable$1.subscribe({
      next(data) {
        isDone = true;
        resolve(data);
        onDone();
      },
      error(data) {
        reject(data);
      },
      complete() {
        ac.abort();
        onDone();
      }
    });
  });
  return promise;
}

// node_modules/@trpc/server/dist/observable-CUiPknO-.mjs
function share(_opts) {
  return (source) => {
    let refCount = 0;
    let subscription = null;
    const observers = [];
    function startIfNeeded() {
      if (subscription)
        return;
      subscription = source.subscribe({
        next(value) {
          for (const observer of observers) {
            var _observer$next;
            (_observer$next = observer.next) === null || _observer$next === undefined || _observer$next.call(observer, value);
          }
        },
        error(error) {
          for (const observer of observers) {
            var _observer$error;
            (_observer$error = observer.error) === null || _observer$error === undefined || _observer$error.call(observer, error);
          }
        },
        complete() {
          for (const observer of observers) {
            var _observer$complete;
            (_observer$complete = observer.complete) === null || _observer$complete === undefined || _observer$complete.call(observer);
          }
        }
      });
    }
    function resetIfNeeded() {
      if (refCount === 0 && subscription) {
        const _sub = subscription;
        subscription = null;
        _sub.unsubscribe();
      }
    }
    return observable((subscriber) => {
      refCount++;
      observers.push(subscriber);
      startIfNeeded();
      return { unsubscribe() {
        refCount--;
        resetIfNeeded();
        const index = observers.findIndex((v) => v === subscriber);
        if (index > -1)
          observers.splice(index, 1);
      } };
    });
  };
}
var distinctUnsetMarker = Symbol();
function behaviorSubject(initialValue) {
  let value = initialValue;
  const observerList = [];
  const addObserver = (observer) => {
    if (value !== undefined)
      observer.next(value);
    observerList.push(observer);
  };
  const removeObserver = (observer) => {
    observerList.splice(observerList.indexOf(observer), 1);
  };
  const obs = observable((observer) => {
    addObserver(observer);
    return () => {
      removeObserver(observer);
    };
  });
  obs.next = (nextValue) => {
    if (value === nextValue)
      return;
    value = nextValue;
    for (const observer of observerList)
      observer.next(nextValue);
  };
  obs.get = () => value;
  return obs;
}

// node_modules/@trpc/client/dist/splitLink-B7Cuf2c_.mjs
function createChain(opts) {
  return observable((observer) => {
    function execute(index = 0, op = opts.op) {
      const next = opts.links[index];
      if (!next)
        throw new Error("No more links to execute - did you forget to add an ending link?");
      const subscription = next({
        op,
        next(nextOp) {
          const nextObserver = execute(index + 1, nextOp);
          return nextObserver;
        }
      });
      return subscription;
    }
    const obs$ = execute();
    return obs$.subscribe(observer);
  });
}
function asArray(value) {
  return Array.isArray(value) ? value : [value];
}
function splitLink(opts) {
  return (runtime) => {
    const yes = asArray(opts.true).map((link) => link(runtime));
    const no = asArray(opts.false).map((link) => link(runtime));
    return (props) => {
      return observable((observer) => {
        const links = opts.condition(props.op) ? yes : no;
        return createChain({
          op: props.op,
          links
        }).subscribe(observer);
      });
    };
  };
}

// node_modules/@trpc/server/dist/codes-DagpWZLc.mjs
function isObject(value) {
  return !!value && !Array.isArray(value) && typeof value === "object";
}
function emptyObject() {
  return Object.create(null);
}
var TRPC_ERROR_CODES_BY_KEY = {
  PARSE_ERROR: -32700,
  BAD_REQUEST: -32600,
  INTERNAL_SERVER_ERROR: -32603,
  NOT_IMPLEMENTED: -32603,
  BAD_GATEWAY: -32603,
  SERVICE_UNAVAILABLE: -32603,
  GATEWAY_TIMEOUT: -32603,
  UNAUTHORIZED: -32001,
  PAYMENT_REQUIRED: -32002,
  FORBIDDEN: -32003,
  NOT_FOUND: -32004,
  METHOD_NOT_SUPPORTED: -32005,
  TIMEOUT: -32008,
  CONFLICT: -32009,
  PRECONDITION_FAILED: -32012,
  PAYLOAD_TOO_LARGE: -32013,
  UNSUPPORTED_MEDIA_TYPE: -32015,
  UNPROCESSABLE_CONTENT: -32022,
  PRECONDITION_REQUIRED: -32028,
  TOO_MANY_REQUESTS: -32029,
  CLIENT_CLOSED_REQUEST: -32099
};
var retryableRpcCodes = [
  TRPC_ERROR_CODES_BY_KEY.BAD_GATEWAY,
  TRPC_ERROR_CODES_BY_KEY.SERVICE_UNAVAILABLE,
  TRPC_ERROR_CODES_BY_KEY.GATEWAY_TIMEOUT,
  TRPC_ERROR_CODES_BY_KEY.INTERNAL_SERVER_ERROR
];

// node_modules/@trpc/server/dist/getErrorShape-vC8mUXJD.mjs
var __create2 = Object.create;
var __defProp2 = Object.defineProperty;
var __getOwnPropDesc2 = Object.getOwnPropertyDescriptor;
var __getOwnPropNames2 = Object.getOwnPropertyNames;
var __getProtoOf2 = Object.getPrototypeOf;
var __hasOwnProp2 = Object.prototype.hasOwnProperty;
var __commonJS2 = (cb, mod) => function() {
  return mod || (0, cb[__getOwnPropNames2(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
};
var __copyProps2 = (to, from, except, desc) => {
  if (from && typeof from === "object" || typeof from === "function")
    for (var keys = __getOwnPropNames2(from), i = 0, n = keys.length, key;i < n; i++) {
      key = keys[i];
      if (!__hasOwnProp2.call(to, key) && key !== except)
        __defProp2(to, key, {
          get: ((k) => from[k]).bind(null, key),
          enumerable: !(desc = __getOwnPropDesc2(from, key)) || desc.enumerable
        });
    }
  return to;
};
var __toESM2 = (mod, isNodeMode, target) => (target = mod != null ? __create2(__getProtoOf2(mod)) : {}, __copyProps2(isNodeMode || !mod || !mod.__esModule ? __defProp2(target, "default", {
  value: mod,
  enumerable: true
}) : target, mod));
var noop = () => {};
var freezeIfAvailable = (obj) => {
  if (Object.freeze)
    Object.freeze(obj);
};
function createInnerProxy(callback, path, memo) {
  var _memo$cacheKey;
  const cacheKey = path.join(".");
  (_memo$cacheKey = memo[cacheKey]) !== null && _memo$cacheKey !== undefined || (memo[cacheKey] = new Proxy(noop, {
    get(_obj, key) {
      if (typeof key !== "string" || key === "then")
        return;
      return createInnerProxy(callback, [...path, key], memo);
    },
    apply(_1, _2, args) {
      const lastOfPath = path[path.length - 1];
      let opts = {
        args,
        path
      };
      if (lastOfPath === "call")
        opts = {
          args: args.length >= 2 ? [args[1]] : [],
          path: path.slice(0, -1)
        };
      else if (lastOfPath === "apply")
        opts = {
          args: args.length >= 2 ? args[1] : [],
          path: path.slice(0, -1)
        };
      freezeIfAvailable(opts.args);
      freezeIfAvailable(opts.path);
      return callback(opts);
    }
  }));
  return memo[cacheKey];
}
var createRecursiveProxy = (callback) => createInnerProxy(callback, [], emptyObject());
var createFlatProxy = (callback) => {
  return new Proxy(noop, { get(_obj, name) {
    if (name === "then")
      return;
    return callback(name);
  } });
};
var require_typeof2 = __commonJS2({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/typeof.js"(exports, module) {
  function _typeof$2(o) {
    "@babel/helpers - typeof";
    return module.exports = _typeof$2 = typeof Symbol == "function" && typeof Symbol.iterator == "symbol" ? function(o$1) {
      return typeof o$1;
    } : function(o$1) {
      return o$1 && typeof Symbol == "function" && o$1.constructor === Symbol && o$1 !== Symbol.prototype ? "symbol" : typeof o$1;
    }, module.exports.__esModule = true, module.exports["default"] = module.exports, _typeof$2(o);
  }
  module.exports = _typeof$2, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_toPrimitive2 = __commonJS2({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/toPrimitive.js"(exports, module) {
  var _typeof$1 = require_typeof2()["default"];
  function toPrimitive$1(t, r) {
    if (_typeof$1(t) != "object" || !t)
      return t;
    var e = t[Symbol.toPrimitive];
    if (e !== undefined) {
      var i = e.call(t, r || "default");
      if (_typeof$1(i) != "object")
        return i;
      throw new TypeError("@@toPrimitive must return a primitive value.");
    }
    return (r === "string" ? String : Number)(t);
  }
  module.exports = toPrimitive$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_toPropertyKey2 = __commonJS2({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/toPropertyKey.js"(exports, module) {
  var _typeof = require_typeof2()["default"];
  var toPrimitive = require_toPrimitive2();
  function toPropertyKey$1(t) {
    var i = toPrimitive(t, "string");
    return _typeof(i) == "symbol" ? i : i + "";
  }
  module.exports = toPropertyKey$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_defineProperty2 = __commonJS2({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/defineProperty.js"(exports, module) {
  var toPropertyKey = require_toPropertyKey2();
  function _defineProperty(e, r, t) {
    return (r = toPropertyKey(r)) in e ? Object.defineProperty(e, r, {
      value: t,
      enumerable: true,
      configurable: true,
      writable: true
    }) : e[r] = t, e;
  }
  module.exports = _defineProperty, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_objectSpread22 = __commonJS2({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/objectSpread2.js"(exports, module) {
  var defineProperty = require_defineProperty2();
  function ownKeys(e, r) {
    var t = Object.keys(e);
    if (Object.getOwnPropertySymbols) {
      var o = Object.getOwnPropertySymbols(e);
      r && (o = o.filter(function(r$1) {
        return Object.getOwnPropertyDescriptor(e, r$1).enumerable;
      })), t.push.apply(t, o);
    }
    return t;
  }
  function _objectSpread2(e) {
    for (var r = 1;r < arguments.length; r++) {
      var t = arguments[r] != null ? arguments[r] : {};
      r % 2 ? ownKeys(Object(t), true).forEach(function(r$1) {
        defineProperty(e, r$1, t[r$1]);
      }) : Object.getOwnPropertyDescriptors ? Object.defineProperties(e, Object.getOwnPropertyDescriptors(t)) : ownKeys(Object(t)).forEach(function(r$1) {
        Object.defineProperty(e, r$1, Object.getOwnPropertyDescriptor(t, r$1));
      });
    }
    return e;
  }
  module.exports = _objectSpread2, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var import_objectSpread2 = __toESM2(require_objectSpread22(), 1);

// node_modules/@trpc/server/dist/tracked-DiE3uR1B.mjs
var import_defineProperty = __toESM2(require_defineProperty2(), 1);
var import_objectSpread2$1 = __toESM2(require_objectSpread22(), 1);
function transformResultInner(response, transformer) {
  if ("error" in response) {
    const error = transformer.deserialize(response.error);
    return {
      ok: false,
      error: (0, import_objectSpread2$1.default)((0, import_objectSpread2$1.default)({}, response), {}, { error })
    };
  }
  const result = (0, import_objectSpread2$1.default)((0, import_objectSpread2$1.default)({}, response.result), (!response.result.type || response.result.type === "data") && {
    type: "data",
    data: transformer.deserialize(response.result.data)
  });
  return {
    ok: true,
    result
  };
}
var TransformResultError = class extends Error {
  constructor() {
    super("Unable to transform response from server");
  }
};
function transformResult(response, transformer) {
  let result;
  try {
    result = transformResultInner(response, transformer);
  } catch (_unused) {
    throw new TransformResultError;
  }
  if (!result.ok && (!isObject(result.error.error) || typeof result.error.error["code"] !== "number"))
    throw new TransformResultError;
  if (result.ok && !isObject(result.result))
    throw new TransformResultError;
  return result;
}
var import_objectSpread22 = __toESM2(require_objectSpread22(), 1);
var trackedSymbol = Symbol();

// node_modules/@trpc/client/dist/TRPCClientError-apv8gw59.mjs
var import_defineProperty2 = __toESM(require_defineProperty(), 1);
var import_objectSpread23 = __toESM(require_objectSpread2(), 1);
function isTRPCClientError(cause) {
  return cause instanceof TRPCClientError;
}
function isTRPCErrorResponse(obj) {
  return isObject(obj) && isObject(obj["error"]) && typeof obj["error"]["code"] === "number" && typeof obj["error"]["message"] === "string";
}
function getMessageFromUnknownError(err, fallback) {
  if (typeof err === "string")
    return err;
  if (isObject(err) && typeof err["message"] === "string")
    return err["message"];
  return fallback;
}
var TRPCClientError = class TRPCClientError2 extends Error {
  constructor(message, opts) {
    var _opts$result, _opts$result2;
    const cause = opts === null || opts === undefined ? undefined : opts.cause;
    super(message, { cause });
    (0, import_defineProperty2.default)(this, "cause", undefined);
    (0, import_defineProperty2.default)(this, "shape", undefined);
    (0, import_defineProperty2.default)(this, "data", undefined);
    (0, import_defineProperty2.default)(this, "meta", undefined);
    this.meta = opts === null || opts === undefined ? undefined : opts.meta;
    this.cause = cause;
    this.shape = opts === null || opts === undefined || (_opts$result = opts.result) === null || _opts$result === undefined ? undefined : _opts$result.error;
    this.data = opts === null || opts === undefined || (_opts$result2 = opts.result) === null || _opts$result2 === undefined ? undefined : _opts$result2.error.data;
    this.name = "TRPCClientError";
    Object.setPrototypeOf(this, TRPCClientError2.prototype);
  }
  static from(_cause, opts = {}) {
    const cause = _cause;
    if (isTRPCClientError(cause)) {
      if (opts.meta)
        cause.meta = (0, import_objectSpread23.default)((0, import_objectSpread23.default)({}, cause.meta), opts.meta);
      return cause;
    }
    if (isTRPCErrorResponse(cause))
      return new TRPCClientError2(cause.error.message, (0, import_objectSpread23.default)((0, import_objectSpread23.default)({}, opts), {}, {
        result: cause,
        cause: opts.cause
      }));
    return new TRPCClientError2(getMessageFromUnknownError(cause, "Unknown error"), (0, import_objectSpread23.default)((0, import_objectSpread23.default)({}, opts), {}, { cause }));
  }
};

// node_modules/@trpc/client/dist/unstable-internals-Bg7n9BBj.mjs
function getTransformer(transformer) {
  const _transformer = transformer;
  if (!_transformer)
    return {
      input: {
        serialize: (data) => data,
        deserialize: (data) => data
      },
      output: {
        serialize: (data) => data,
        deserialize: (data) => data
      }
    };
  if ("input" in _transformer)
    return _transformer;
  return {
    input: _transformer,
    output: _transformer
  };
}

// node_modules/@trpc/client/dist/httpUtils-BNq9QC3d.mjs
var isFunction2 = (fn) => typeof fn === "function";
function getFetch(customFetchImpl) {
  if (customFetchImpl)
    return customFetchImpl;
  if (typeof window !== "undefined" && isFunction2(window.fetch))
    return window.fetch;
  if (typeof globalThis !== "undefined" && isFunction2(globalThis.fetch))
    return globalThis.fetch;
  throw new Error("No fetch implementation found");
}
var import_objectSpread24 = __toESM(require_objectSpread2());
function resolveHTTPLinkOptions(opts) {
  return {
    url: opts.url.toString(),
    fetch: opts.fetch,
    transformer: getTransformer(opts.transformer),
    methodOverride: opts.methodOverride
  };
}
function arrayToDict(array) {
  const dict = {};
  for (let index = 0;index < array.length; index++) {
    const element = array[index];
    dict[index] = element;
  }
  return dict;
}
var METHOD = {
  query: "GET",
  mutation: "POST",
  subscription: "PATCH"
};
function getInput(opts) {
  return "input" in opts ? opts.transformer.input.serialize(opts.input) : arrayToDict(opts.inputs.map((_input) => opts.transformer.input.serialize(_input)));
}
var getUrl = (opts) => {
  const parts = opts.url.split("?");
  const base = parts[0].replace(/\/$/, "");
  let url = base + "/" + opts.path;
  const queryParts = [];
  if (parts[1])
    queryParts.push(parts[1]);
  if ("inputs" in opts)
    queryParts.push("batch=1");
  if (opts.type === "query" || opts.type === "subscription") {
    const input = getInput(opts);
    if (input !== undefined && opts.methodOverride !== "POST")
      queryParts.push(`input=${encodeURIComponent(JSON.stringify(input))}`);
  }
  if (queryParts.length)
    url += "?" + queryParts.join("&");
  return url;
};
var getBody = (opts) => {
  if (opts.type === "query" && opts.methodOverride !== "POST")
    return;
  const input = getInput(opts);
  return input !== undefined ? JSON.stringify(input) : undefined;
};
var jsonHttpRequester = (opts) => {
  return httpRequest((0, import_objectSpread24.default)((0, import_objectSpread24.default)({}, opts), {}, {
    contentTypeHeader: "application/json",
    getUrl,
    getBody
  }));
};
var AbortError = class extends Error {
  constructor() {
    const name = "AbortError";
    super(name);
    this.name = name;
    this.message = name;
  }
};
var throwIfAborted = (signal) => {
  var _signal$throwIfAborte;
  if (!(signal === null || signal === undefined ? undefined : signal.aborted))
    return;
  (_signal$throwIfAborte = signal.throwIfAborted) === null || _signal$throwIfAborte === undefined || _signal$throwIfAborte.call(signal);
  if (typeof DOMException !== "undefined")
    throw new DOMException("AbortError", "AbortError");
  throw new AbortError;
};
async function fetchHTTPResponse(opts) {
  var _opts$methodOverride, _opts$trpcAcceptHeade;
  throwIfAborted(opts.signal);
  const url = opts.getUrl(opts);
  const body = opts.getBody(opts);
  const method = (_opts$methodOverride = opts.methodOverride) !== null && _opts$methodOverride !== undefined ? _opts$methodOverride : METHOD[opts.type];
  const resolvedHeaders = await (async () => {
    const heads = await opts.headers();
    if (Symbol.iterator in heads)
      return Object.fromEntries(heads);
    return heads;
  })();
  const headers = (0, import_objectSpread24.default)((0, import_objectSpread24.default)((0, import_objectSpread24.default)({}, opts.contentTypeHeader && method !== "GET" ? { "content-type": opts.contentTypeHeader } : {}), opts.trpcAcceptHeader ? { [(_opts$trpcAcceptHeade = opts.trpcAcceptHeaderKey) !== null && _opts$trpcAcceptHeade !== undefined ? _opts$trpcAcceptHeade : "trpc-accept"]: opts.trpcAcceptHeader } : undefined), resolvedHeaders);
  return getFetch(opts.fetch)(url, {
    method,
    signal: opts.signal,
    body,
    headers
  });
}
async function httpRequest(opts) {
  const meta = {};
  const res = await fetchHTTPResponse(opts);
  meta.response = res;
  const json = await res.json();
  meta.responseJSON = json;
  return {
    json,
    meta
  };
}

// node_modules/@trpc/client/dist/httpLink-oiU8eqFi.mjs
function isOctetType(input) {
  return input instanceof Uint8Array || input instanceof Blob;
}
function isFormData(input) {
  return input instanceof FormData;
}
var import_objectSpread25 = __toESM(require_objectSpread2(), 1);
var universalRequester = (opts) => {
  if ("input" in opts) {
    const { input } = opts;
    if (isFormData(input)) {
      if (opts.type !== "mutation" && opts.methodOverride !== "POST")
        throw new Error("FormData is only supported for mutations");
      return httpRequest((0, import_objectSpread25.default)((0, import_objectSpread25.default)({}, opts), {}, {
        contentTypeHeader: undefined,
        getUrl,
        getBody: () => input
      }));
    }
    if (isOctetType(input)) {
      if (opts.type !== "mutation" && opts.methodOverride !== "POST")
        throw new Error("Octet type input is only supported for mutations");
      return httpRequest((0, import_objectSpread25.default)((0, import_objectSpread25.default)({}, opts), {}, {
        contentTypeHeader: "application/octet-stream",
        getUrl,
        getBody: () => input
      }));
    }
  }
  return jsonHttpRequester(opts);
};
function httpLink(opts) {
  const resolvedOpts = resolveHTTPLinkOptions(opts);
  return () => {
    return (operationOpts) => {
      const { op } = operationOpts;
      return observable((observer) => {
        const { path, input, type } = op;
        if (type === "subscription")
          throw new Error("Subscriptions are unsupported by `httpLink` - use `httpSubscriptionLink` or `wsLink`");
        const request = universalRequester((0, import_objectSpread25.default)((0, import_objectSpread25.default)({}, resolvedOpts), {}, {
          type,
          path,
          input,
          signal: op.signal,
          headers() {
            if (!opts.headers)
              return {};
            if (typeof opts.headers === "function")
              return opts.headers({ op });
            return opts.headers;
          }
        }));
        let meta = undefined;
        request.then((res) => {
          meta = res.meta;
          const transformed = transformResult(res.json, resolvedOpts.transformer.output);
          if (!transformed.ok) {
            observer.error(TRPCClientError.from(transformed.error, { meta }));
            return;
          }
          observer.next({
            context: res.meta,
            result: transformed.result
          });
          observer.complete();
        }).catch((cause) => {
          observer.error(TRPCClientError.from(cause, { meta }));
        });
        return () => {};
      });
    };
  };
}

// node_modules/@trpc/client/dist/httpBatchLink-CaWjh1oW.mjs
var throwFatalError = () => {
  throw new Error("Something went wrong. Please submit an issue at https://github.com/trpc/trpc/issues/new");
};
function dataLoader(batchLoader) {
  let pendingItems = null;
  let dispatchTimer = null;
  const destroyTimerAndPendingItems = () => {
    clearTimeout(dispatchTimer);
    dispatchTimer = null;
    pendingItems = null;
  };
  function groupItems(items) {
    const groupedItems = [[]];
    let index = 0;
    while (true) {
      const item = items[index];
      if (!item)
        break;
      const lastGroup = groupedItems[groupedItems.length - 1];
      if (item.aborted) {
        var _item$reject;
        (_item$reject = item.reject) === null || _item$reject === undefined || _item$reject.call(item, new Error("Aborted"));
        index++;
        continue;
      }
      const isValid = batchLoader.validate(lastGroup.concat(item).map((it) => it.key));
      if (isValid) {
        lastGroup.push(item);
        index++;
        continue;
      }
      if (lastGroup.length === 0) {
        var _item$reject2;
        (_item$reject2 = item.reject) === null || _item$reject2 === undefined || _item$reject2.call(item, new Error("Input is too big for a single dispatch"));
        index++;
        continue;
      }
      groupedItems.push([]);
    }
    return groupedItems;
  }
  function dispatch() {
    const groupedItems = groupItems(pendingItems);
    destroyTimerAndPendingItems();
    for (const items of groupedItems) {
      if (!items.length)
        continue;
      const batch = { items };
      for (const item of items)
        item.batch = batch;
      const promise = batchLoader.fetch(batch.items.map((_item) => _item.key));
      promise.then(async (result) => {
        await Promise.all(result.map(async (valueOrPromise, index) => {
          const item = batch.items[index];
          try {
            var _item$resolve;
            const value = await Promise.resolve(valueOrPromise);
            (_item$resolve = item.resolve) === null || _item$resolve === undefined || _item$resolve.call(item, value);
          } catch (cause) {
            var _item$reject3;
            (_item$reject3 = item.reject) === null || _item$reject3 === undefined || _item$reject3.call(item, cause);
          }
          item.batch = null;
          item.reject = null;
          item.resolve = null;
        }));
        for (const item of batch.items) {
          var _item$reject4;
          (_item$reject4 = item.reject) === null || _item$reject4 === undefined || _item$reject4.call(item, new Error("Missing result"));
          item.batch = null;
        }
      }).catch((cause) => {
        for (const item of batch.items) {
          var _item$reject5;
          (_item$reject5 = item.reject) === null || _item$reject5 === undefined || _item$reject5.call(item, cause);
          item.batch = null;
        }
      });
    }
  }
  function load(key) {
    var _dispatchTimer;
    const item = {
      aborted: false,
      key,
      batch: null,
      resolve: throwFatalError,
      reject: throwFatalError
    };
    const promise = new Promise((resolve, reject) => {
      var _pendingItems;
      item.reject = reject;
      item.resolve = resolve;
      (_pendingItems = pendingItems) !== null && _pendingItems !== undefined || (pendingItems = []);
      pendingItems.push(item);
    });
    (_dispatchTimer = dispatchTimer) !== null && _dispatchTimer !== undefined || (dispatchTimer = setTimeout(dispatch));
    return promise;
  }
  return { load };
}
function allAbortSignals(...signals) {
  const ac = new AbortController;
  const count = signals.length;
  let abortedCount = 0;
  const onAbort = () => {
    if (++abortedCount === count)
      ac.abort();
  };
  for (const signal of signals)
    if (signal === null || signal === undefined ? undefined : signal.aborted)
      onAbort();
    else
      signal === null || signal === undefined || signal.addEventListener("abort", onAbort, { once: true });
  return ac.signal;
}
var import_objectSpread26 = __toESM(require_objectSpread2(), 1);
function httpBatchLink(opts) {
  var _opts$maxURLLength, _opts$maxItems;
  const resolvedOpts = resolveHTTPLinkOptions(opts);
  const maxURLLength = (_opts$maxURLLength = opts.maxURLLength) !== null && _opts$maxURLLength !== undefined ? _opts$maxURLLength : Infinity;
  const maxItems = (_opts$maxItems = opts.maxItems) !== null && _opts$maxItems !== undefined ? _opts$maxItems : Infinity;
  return () => {
    const batchLoader = (type) => {
      return {
        validate(batchOps) {
          if (maxURLLength === Infinity && maxItems === Infinity)
            return true;
          if (batchOps.length > maxItems)
            return false;
          const path = batchOps.map((op) => op.path).join(",");
          const inputs = batchOps.map((op) => op.input);
          const url = getUrl((0, import_objectSpread26.default)((0, import_objectSpread26.default)({}, resolvedOpts), {}, {
            type,
            path,
            inputs,
            signal: null
          }));
          return url.length <= maxURLLength;
        },
        async fetch(batchOps) {
          const path = batchOps.map((op) => op.path).join(",");
          const inputs = batchOps.map((op) => op.input);
          const signal = allAbortSignals(...batchOps.map((op) => op.signal));
          const res = await jsonHttpRequester((0, import_objectSpread26.default)((0, import_objectSpread26.default)({}, resolvedOpts), {}, {
            path,
            inputs,
            type,
            headers() {
              if (!opts.headers)
                return {};
              if (typeof opts.headers === "function")
                return opts.headers({ opList: batchOps });
              return opts.headers;
            },
            signal
          }));
          const resJSON = Array.isArray(res.json) ? res.json : batchOps.map(() => res.json);
          const result = resJSON.map((item) => ({
            meta: res.meta,
            json: item
          }));
          return result;
        }
      };
    };
    const query = dataLoader(batchLoader("query"));
    const mutation = dataLoader(batchLoader("mutation"));
    const loaders = {
      query,
      mutation
    };
    return ({ op }) => {
      return observable((observer) => {
        if (op.type === "subscription")
          throw new Error("Subscriptions are unsupported by `httpLink` - use `httpSubscriptionLink` or `wsLink`");
        const loader = loaders[op.type];
        const promise = loader.load(op);
        let _res = undefined;
        promise.then((res) => {
          _res = res;
          const transformed = transformResult(res.json, resolvedOpts.transformer.output);
          if (!transformed.ok) {
            observer.error(TRPCClientError.from(transformed.error, { meta: res.meta }));
            return;
          }
          observer.next({
            context: res.meta,
            result: transformed.result
          });
          observer.complete();
        }).catch((err) => {
          observer.error(TRPCClientError.from(err, { meta: _res === null || _res === undefined ? undefined : _res.meta }));
        });
        return () => {};
      });
    };
  };
}

// node_modules/@trpc/client/dist/loggerLink-ineCN1PO.mjs
var import_objectSpread27 = __toESM(require_objectSpread2(), 1);

// node_modules/@trpc/client/dist/wsLink-DSf4KOdW.mjs
var resultOf = (value, ...args) => {
  return typeof value === "function" ? value(...args) : value;
};
var import_defineProperty$3 = __toESM(require_defineProperty(), 1);
function withResolvers() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return {
    promise,
    resolve,
    reject
  };
}
async function prepareUrl(urlOptions) {
  const url = await resultOf(urlOptions.url);
  if (!urlOptions.connectionParams)
    return url;
  const prefix = url.includes("?") ? "&" : "?";
  const connectionParams = `${prefix}connectionParams=1`;
  return url + connectionParams;
}
async function buildConnectionMessage(connectionParams, encoder) {
  const message = {
    method: "connectionParams",
    data: await resultOf(connectionParams)
  };
  return encoder.encode(message);
}
var import_defineProperty$2 = __toESM(require_defineProperty(), 1);
var import_defineProperty$1 = __toESM(require_defineProperty(), 1);
function asyncWsOpen(ws) {
  const { promise, resolve, reject } = withResolvers();
  ws.addEventListener("open", () => {
    ws.removeEventListener("error", reject);
    resolve();
  });
  ws.addEventListener("error", reject);
  return promise;
}
function setupPingInterval(ws, { intervalMs, pongTimeoutMs }) {
  let pingTimeout;
  let pongTimeout;
  function start() {
    pingTimeout = setTimeout(() => {
      ws.send("PING");
      pongTimeout = setTimeout(() => {
        ws.close();
      }, pongTimeoutMs);
    }, intervalMs);
  }
  function reset() {
    clearTimeout(pingTimeout);
    start();
  }
  function pong() {
    clearTimeout(pongTimeout);
    reset();
  }
  ws.addEventListener("open", start);
  ws.addEventListener("message", ({ data }) => {
    clearTimeout(pingTimeout);
    start();
    if (data === "PONG")
      pong();
  });
  ws.addEventListener("close", () => {
    clearTimeout(pingTimeout);
    clearTimeout(pongTimeout);
  });
}
var WsConnection = class WsConnection2 {
  constructor(opts) {
    var _opts$WebSocketPonyfi;
    (0, import_defineProperty$1.default)(this, "id", ++WsConnection2.connectCount);
    (0, import_defineProperty$1.default)(this, "WebSocketPonyfill", undefined);
    (0, import_defineProperty$1.default)(this, "urlOptions", undefined);
    (0, import_defineProperty$1.default)(this, "keepAliveOpts", undefined);
    (0, import_defineProperty$1.default)(this, "encoder", undefined);
    (0, import_defineProperty$1.default)(this, "wsObservable", behaviorSubject(null));
    (0, import_defineProperty$1.default)(this, "openPromise", null);
    this.WebSocketPonyfill = (_opts$WebSocketPonyfi = opts.WebSocketPonyfill) !== null && _opts$WebSocketPonyfi !== undefined ? _opts$WebSocketPonyfi : WebSocket;
    if (!this.WebSocketPonyfill)
      throw new Error("No WebSocket implementation found - you probably don't want to use this on the server, but if you do you need to pass a `WebSocket`-ponyfill");
    this.urlOptions = opts.urlOptions;
    this.keepAliveOpts = opts.keepAlive;
    this.encoder = opts.encoder;
  }
  get ws() {
    return this.wsObservable.get();
  }
  set ws(ws) {
    this.wsObservable.next(ws);
  }
  isOpen() {
    return !!this.ws && this.ws.readyState === this.WebSocketPonyfill.OPEN && !this.openPromise;
  }
  isClosed() {
    return !!this.ws && (this.ws.readyState === this.WebSocketPonyfill.CLOSING || this.ws.readyState === this.WebSocketPonyfill.CLOSED);
  }
  async open() {
    var _this = this;
    if (_this.openPromise)
      return _this.openPromise;
    _this.id = ++WsConnection2.connectCount;
    const wsPromise = prepareUrl(_this.urlOptions).then((url) => new _this.WebSocketPonyfill(url));
    _this.openPromise = wsPromise.then(async (ws) => {
      _this.ws = ws;
      ws.binaryType = "arraybuffer";
      ws.addEventListener("message", function({ data }) {
        if (data === "PING")
          this.send("PONG");
      });
      if (_this.keepAliveOpts.enabled)
        setupPingInterval(ws, _this.keepAliveOpts);
      ws.addEventListener("close", () => {
        if (_this.ws === ws)
          _this.ws = null;
      });
      await asyncWsOpen(ws);
      if (_this.urlOptions.connectionParams)
        ws.send(await buildConnectionMessage(_this.urlOptions.connectionParams, _this.encoder));
    });
    try {
      await _this.openPromise;
    } finally {
      _this.openPromise = null;
    }
  }
  async close() {
    var _this2 = this;
    try {
      await _this2.openPromise;
    } finally {
      var _this$ws;
      (_this$ws = _this2.ws) === null || _this$ws === undefined || _this$ws.close();
    }
  }
};
(0, import_defineProperty$1.default)(WsConnection, "connectCount", 0);
var import_defineProperty3 = __toESM(require_defineProperty(), 1);
var import_objectSpread28 = __toESM(require_objectSpread2(), 1);

// node_modules/@trpc/client/dist/index.mjs
var import_defineProperty4 = __toESM(require_defineProperty(), 1);
var import_objectSpread2$4 = __toESM(require_objectSpread2(), 1);
var TRPCUntypedClient = class {
  constructor(opts) {
    (0, import_defineProperty4.default)(this, "links", undefined);
    (0, import_defineProperty4.default)(this, "runtime", undefined);
    (0, import_defineProperty4.default)(this, "requestId", undefined);
    this.requestId = 0;
    this.runtime = {};
    this.links = opts.links.map((link) => link(this.runtime));
  }
  $request(opts) {
    var _opts$context;
    const chain$ = createChain({
      links: this.links,
      op: (0, import_objectSpread2$4.default)((0, import_objectSpread2$4.default)({}, opts), {}, {
        context: (_opts$context = opts.context) !== null && _opts$context !== undefined ? _opts$context : {},
        id: ++this.requestId
      })
    });
    return chain$.pipe(share());
  }
  async requestAsPromise(opts) {
    var _this = this;
    try {
      const req$ = _this.$request(opts);
      const envelope = await observableToPromise(req$);
      const data = envelope.result.data;
      return data;
    } catch (err) {
      throw TRPCClientError.from(err);
    }
  }
  query(path, input, opts) {
    return this.requestAsPromise({
      type: "query",
      path,
      input,
      context: opts === null || opts === undefined ? undefined : opts.context,
      signal: opts === null || opts === undefined ? undefined : opts.signal
    });
  }
  mutation(path, input, opts) {
    return this.requestAsPromise({
      type: "mutation",
      path,
      input,
      context: opts === null || opts === undefined ? undefined : opts.context,
      signal: opts === null || opts === undefined ? undefined : opts.signal
    });
  }
  subscription(path, input, opts) {
    const observable$ = this.$request({
      type: "subscription",
      path,
      input,
      context: opts.context,
      signal: opts.signal
    });
    return observable$.subscribe({
      next(envelope) {
        switch (envelope.result.type) {
          case "state": {
            var _opts$onConnectionSta;
            (_opts$onConnectionSta = opts.onConnectionStateChange) === null || _opts$onConnectionSta === undefined || _opts$onConnectionSta.call(opts, envelope.result);
            break;
          }
          case "started": {
            var _opts$onStarted;
            (_opts$onStarted = opts.onStarted) === null || _opts$onStarted === undefined || _opts$onStarted.call(opts, { context: envelope.context });
            break;
          }
          case "stopped": {
            var _opts$onStopped;
            (_opts$onStopped = opts.onStopped) === null || _opts$onStopped === undefined || _opts$onStopped.call(opts);
            break;
          }
          case "data":
          case undefined: {
            var _opts$onData;
            (_opts$onData = opts.onData) === null || _opts$onData === undefined || _opts$onData.call(opts, envelope.result.data);
            break;
          }
        }
      },
      error(err) {
        var _opts$onError;
        (_opts$onError = opts.onError) === null || _opts$onError === undefined || _opts$onError.call(opts, err);
      },
      complete() {
        var _opts$onComplete;
        (_opts$onComplete = opts.onComplete) === null || _opts$onComplete === undefined || _opts$onComplete.call(opts);
      }
    });
  }
};
var untypedClientSymbol = Symbol.for("trpc_untypedClient");
var clientCallTypeMap = {
  query: "query",
  mutate: "mutation",
  subscribe: "subscription"
};
var clientCallTypeToProcedureType = (clientCallType) => {
  return clientCallTypeMap[clientCallType];
};
function createTRPCClientProxy(client) {
  const proxy = createRecursiveProxy(({ path, args }) => {
    const pathCopy = [...path];
    const procedureType = clientCallTypeToProcedureType(pathCopy.pop());
    const fullPath = pathCopy.join(".");
    return client[procedureType](fullPath, ...args);
  });
  return createFlatProxy((key) => {
    if (key === untypedClientSymbol)
      return client;
    return proxy[key];
  });
}
function createTRPCClient(opts) {
  const client = new TRPCUntypedClient(opts);
  const proxy = createTRPCClientProxy(client);
  return proxy;
}
var import_objectSpread2$3 = __toESM(require_objectSpread2(), 1);
var import_objectSpread2$2 = __toESM(require_objectSpread2(), 1);
var require_asyncIterator = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/asyncIterator.js"(exports, module) {
  function _asyncIterator$1(r) {
    var n, t, o, e = 2;
    for (typeof Symbol != "undefined" && (t = Symbol.asyncIterator, o = Symbol.iterator);e--; ) {
      if (t && (n = r[t]) != null)
        return n.call(r);
      if (o && (n = r[o]) != null)
        return new AsyncFromSyncIterator(n.call(r));
      t = "@@asyncIterator", o = "@@iterator";
    }
    throw new TypeError("Object is not async iterable");
  }
  function AsyncFromSyncIterator(r) {
    function AsyncFromSyncIteratorContinuation(r$1) {
      if (Object(r$1) !== r$1)
        return Promise.reject(new TypeError(r$1 + " is not an object."));
      var n = r$1.done;
      return Promise.resolve(r$1.value).then(function(r$2) {
        return {
          value: r$2,
          done: n
        };
      });
    }
    return AsyncFromSyncIterator = function AsyncFromSyncIterator$1(r$1) {
      this.s = r$1, this.n = r$1.next;
    }, AsyncFromSyncIterator.prototype = {
      s: null,
      n: null,
      next: function next() {
        return AsyncFromSyncIteratorContinuation(this.n.apply(this.s, arguments));
      },
      return: function _return(r$1) {
        var n = this.s["return"];
        return n === undefined ? Promise.resolve({
          value: r$1,
          done: true
        }) : AsyncFromSyncIteratorContinuation(n.apply(this.s, arguments));
      },
      throw: function _throw(r$1) {
        var n = this.s["return"];
        return n === undefined ? Promise.reject(r$1) : AsyncFromSyncIteratorContinuation(n.apply(this.s, arguments));
      }
    }, new AsyncFromSyncIterator(r);
  }
  module.exports = _asyncIterator$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var import_asyncIterator = __toESM(require_asyncIterator(), 1);
var import_objectSpread2$12 = __toESM(require_objectSpread2(), 1);
var require_usingCtx = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/usingCtx.js"(exports, module) {
  function _usingCtx() {
    var r = typeof SuppressedError == "function" ? SuppressedError : function(r$1, e$1) {
      var n$1 = Error();
      return n$1.name = "SuppressedError", n$1.error = r$1, n$1.suppressed = e$1, n$1;
    }, e = {}, n = [];
    function using(r$1, e$1) {
      if (e$1 != null) {
        if (Object(e$1) !== e$1)
          throw new TypeError("using declarations can only be used with objects, functions, null, or undefined.");
        if (r$1)
          var o = e$1[Symbol.asyncDispose || Symbol["for"]("Symbol.asyncDispose")];
        if (o === undefined && (o = e$1[Symbol.dispose || Symbol["for"]("Symbol.dispose")], r$1))
          var t = o;
        if (typeof o != "function")
          throw new TypeError("Object is not disposable.");
        t && (o = function o$1() {
          try {
            t.call(e$1);
          } catch (r$2) {
            return Promise.reject(r$2);
          }
        }), n.push({
          v: e$1,
          d: o,
          a: r$1
        });
      } else
        r$1 && n.push({
          d: e$1,
          a: r$1
        });
      return e$1;
    }
    return {
      e,
      u: using.bind(null, false),
      a: using.bind(null, true),
      d: function d() {
        var o, t = this.e, s = 0;
        function next() {
          for (;o = n.pop(); )
            try {
              if (!o.a && s === 1)
                return s = 0, n.push(o), Promise.resolve().then(next);
              if (o.d) {
                var r$1 = o.d.call(o.v);
                if (o.a)
                  return s |= 2, Promise.resolve(r$1).then(next, err);
              } else
                s |= 1;
            } catch (r$2) {
              return err(r$2);
            }
          if (s === 1)
            return t !== e ? Promise.reject(t) : Promise.resolve();
          if (t !== e)
            throw t;
        }
        function err(n$1) {
          return t = t !== e ? new r(n$1, t) : n$1, next();
        }
        return next();
      }
    };
  }
  module.exports = _usingCtx, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_OverloadYield = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/OverloadYield.js"(exports, module) {
  function _OverloadYield(e, d) {
    this.v = e, this.k = d;
  }
  module.exports = _OverloadYield, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_awaitAsyncGenerator = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/awaitAsyncGenerator.js"(exports, module) {
  var OverloadYield$1 = require_OverloadYield();
  function _awaitAsyncGenerator$1(e) {
    return new OverloadYield$1(e, 0);
  }
  module.exports = _awaitAsyncGenerator$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var require_wrapAsyncGenerator = __commonJS({ "../../node_modules/.pnpm/@oxc-project+runtime@0.72.2/node_modules/@oxc-project/runtime/src/helpers/wrapAsyncGenerator.js"(exports, module) {
  var OverloadYield = require_OverloadYield();
  function _wrapAsyncGenerator$1(e) {
    return function() {
      return new AsyncGenerator(e.apply(this, arguments));
    };
  }
  function AsyncGenerator(e) {
    var r, t;
    function resume(r$1, t$1) {
      try {
        var n = e[r$1](t$1), o = n.value, u = o instanceof OverloadYield;
        Promise.resolve(u ? o.v : o).then(function(t$2) {
          if (u) {
            var i = r$1 === "return" ? "return" : "next";
            if (!o.k || t$2.done)
              return resume(i, t$2);
            t$2 = e[i](t$2).value;
          }
          settle(n.done ? "return" : "normal", t$2);
        }, function(e$1) {
          resume("throw", e$1);
        });
      } catch (e$1) {
        settle("throw", e$1);
      }
    }
    function settle(e$1, n) {
      switch (e$1) {
        case "return":
          r.resolve({
            value: n,
            done: true
          });
          break;
        case "throw":
          r.reject(n);
          break;
        default:
          r.resolve({
            value: n,
            done: false
          });
      }
      (r = r.next) ? resume(r.key, r.arg) : t = null;
    }
    this._invoke = function(e$1, n) {
      return new Promise(function(o, u) {
        var i = {
          key: e$1,
          arg: n,
          resolve: o,
          reject: u,
          next: null
        };
        t ? t = t.next = i : (r = t = i, resume(e$1, n));
      });
    }, typeof e["return"] != "function" && (this["return"] = undefined);
  }
  AsyncGenerator.prototype[typeof Symbol == "function" && Symbol.asyncIterator || "@@asyncIterator"] = function() {
    return this;
  }, AsyncGenerator.prototype.next = function(e) {
    return this._invoke("next", e);
  }, AsyncGenerator.prototype["throw"] = function(e) {
    return this._invoke("throw", e);
  }, AsyncGenerator.prototype["return"] = function(e) {
    return this._invoke("return", e);
  };
  module.exports = _wrapAsyncGenerator$1, module.exports.__esModule = true, module.exports["default"] = module.exports;
} });
var import_usingCtx = __toESM(require_usingCtx(), 1);
var import_awaitAsyncGenerator = __toESM(require_awaitAsyncGenerator(), 1);
var import_wrapAsyncGenerator = __toESM(require_wrapAsyncGenerator(), 1);
var import_objectSpread29 = __toESM(require_objectSpread2(), 1);

// node_modules/@warmhub/sdk-ts/dist/chunk-ATBJRTIK.js
var DEFAULT_INGEST_CHUNK_SIZE = DEFAULT_INGEST_APPEND_CHUNK_SIZE;
async function submitCommitViaIngest(client, args) {
  const chunkSize = normalizeChunkSize(args.chunkSize);
  const beginResult = await client.ingest.begin({
    orgName: args.orgName,
    repoName: args.repoName,
    committer: args.committer,
    message: args.message,
    workspaceName: args.workspaceName,
    componentId: args.componentId
  });
  let cursor = beginResult.cursor;
  const chunkResults = [];
  for (const chunk of chunkOperations(args.operations, chunkSize)) {
    const appendResult = await client.ingest.append({
      orgName: args.orgName,
      repoName: args.repoName,
      ingestId: beginResult.ingestId,
      cursor,
      operations: chunk
    });
    if (!appendResult.deduplicated) {
      chunkResults.push(...appendResult.results);
    }
    cursor = appendResult.nextCursor;
  }
  const finalizeResult = await client.ingest.finalize({
    orgName: args.orgName,
    repoName: args.repoName,
    ingestId: beginResult.ingestId
  });
  if (finalizeResult.status === "finalizing") {
    const completedStatus = await waitForIngestCompletion(client, {
      orgName: args.orgName,
      repoName: args.repoName,
      ingestId: beginResult.ingestId
    });
    return toSubmittedCommitResult({
      commitId: completedStatus.resultCommitId,
      operationCount: completedStatus.resultOperationCount ?? args.operations.length
    }, args.committer, args.message, chunkResults);
  }
  return toSubmittedCommitResult({
    commitId: finalizeResult.commitId,
    operationCount: finalizeResult.operationCount
  }, args.committer, args.message, finalizeResult.operations ?? chunkResults);
}
function normalizeChunkSize(chunkSize) {
  if (!Number.isFinite(chunkSize))
    return DEFAULT_INGEST_CHUNK_SIZE;
  return Math.max(1, Math.min(MAX_INGEST_APPEND_OPERATION_COUNT, Math.trunc(chunkSize)));
}
function chunkOperations(operations, chunkSize) {
  const chunks = [];
  for (let start = 0;start < operations.length; start += chunkSize) {
    chunks.push(operations.slice(start, start + chunkSize));
  }
  return chunks;
}
function toSubmittedCommitResult(finalizeResult, committer, message, results) {
  if (!finalizeResult.commitId) {
    throw new Error("Finalize ingest completed without a commit id");
  }
  return {
    commitId: finalizeResult.commitId,
    committer,
    message,
    operationCount: finalizeResult.operationCount,
    operations: results.map((result) => ({
      name: result.name ?? "",
      operation: result.operation === "revise" || result.operation === "retract" || result.operation === "noop" ? result.operation : "add",
      dataHash: result.dataHash ?? "",
      version: result.version ?? 0
    }))
  };
}
async function waitForIngestCompletion(client, args) {
  for (let attempt = 0;attempt < 1800; attempt += 1) {
    const status = await client.ingest.status(args);
    if (status.status === "completed") {
      return status;
    }
    if (status.status === "failed") {
      throw new Error(status.errorMessage ?? "Ingest finalize failed");
    }
    if (status.status === "aborted") {
      throw new Error("Ingest was aborted");
    }
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error("Timed out waiting for ingest finalization (3 minutes). The ingest may still be processing server-side. Check ingest status or backend logs for details.");
}
var DEFAULT_API_URL = "https://api.warmhub.ai";
var UNBATCHED_TRPC_PATHS = /* @__PURE__ */ new Set(["repo.shapeInstanceCounts"]);
function normalizeSubscriptionInfo(subscription) {
  return {
    ...subscription,
    actionContainer: undefined,
    actionContainerConfig: undefined
  };
}
function normalizeSubscriptionList(subscriptions) {
  return subscriptions.map((subscription) => normalizeSubscriptionInfo(subscription));
}
function sanitizeSubscriptionCreateInput(input) {
  const {
    actionContainer: _actionContainer,
    actionContainerConfig: _actionContainerConfig,
    ...supported
  } = input;
  return supported;
}
function sanitizeSubscriptionUpdateInput(input) {
  const {
    actionContainer: _actionContainer,
    actionContainerConfig: _actionContainerConfig,
    ...supported
  } = input;
  return supported;
}
var INTERNAL_PATTERNS = [
  /\bselect\b.+?\bfrom\b/i,
  /\binsert\b.+?\binto\b/i,
  /\bupdate\b.+?\bset\b/i,
  /\bdelete\b.+?\bfrom\b/i,
  /\bparams:\s/i,
  /\.ts\(\d+,\d+\)/,
  /at\s+\S+\s+\(.*:\d+:\d+\)/,
  /node_modules\//,
  /drizzle/i
];
function sanitizeErrorMessage(message) {
  for (const pattern of INTERNAL_PATTERNS) {
    if (pattern.test(message)) {
      return "An internal error occurred. Please try again or contact support.";
    }
  }
  return message;
}
var WarmHubError = class extends Error {
  code;
  status;
  hint;
  retryAfter;
  constructor(code, message, status, hint, retryAfter) {
    super(message);
    this.name = "WarmHubError";
    this.code = code;
    this.status = status;
    this.hint = hint;
    this.retryAfter = retryAfter;
  }
  get kind() {
    return this.code;
  }
};
function toWarmHubError(error) {
  if (error instanceof WarmHubError) {
    return error;
  }
  if (error instanceof TRPCClientError) {
    if (error.cause instanceof WarmHubError) {
      return error.cause;
    }
    const data = error.data;
    const message = data?.warmhub?.message ?? sanitizeErrorMessage(error.message);
    return new WarmHubError(data?.warmhub?.code ?? "BACKEND", message, data?.warmhub?.status, data?.warmhub?.hint, data?.warmhub?.retryAfter);
  }
  if (error instanceof Error) {
    if (error.name === "AbortError") {
      return new WarmHubError("CANCELLED", error.message);
    }
    if (error instanceof TypeError && /fetch/i.test(error.message)) {
      return new WarmHubError("NETWORK", error.message);
    }
    return new WarmHubError("BACKEND", sanitizeErrorMessage(error.message));
  }
  return new WarmHubError("BACKEND", "Unknown SDK error");
}
var CONNECTION_ERROR_CODES = /* @__PURE__ */ new Set([
  "ConnectionRefused",
  "ECONNREFUSED",
  "ENOTFOUND",
  "ETIMEDOUT",
  "EAI_AGAIN"
]);
function isConnectionError(error) {
  if (error instanceof TypeError && /fetch/i.test(error.message)) {
    return true;
  }
  if (error && typeof error === "object") {
    const code = error.code;
    if (typeof code === "string" && CONNECTION_ERROR_CODES.has(code)) {
      return true;
    }
  }
  return false;
}
function connectionErrorMessage(url) {
  return `Unable to connect to ${url}. Is the computer able to access the url?`;
}
var WarmHubClient = class _WarmHubClient {
  apiUrl;
  fetchImpl;
  accessToken;
  functionLogMode;
  getToken;
  auth = {
    getClientId: async () => {
      try {
        return await this.trpc.auth.getClientId.query();
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    sync: async () => {
      try {
        return await this.trpc.auth.sync.mutate();
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    currentUser: async () => {
      try {
        return await this.trpc.auth.currentUser.query();
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    whoami: async () => {
      try {
        return await this.trpc.auth.whoami.query();
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  access = {
    checkRepoPermission: async (orgName, repoName, permission) => {
      try {
        return await this.trpc.access.checkRepoPermission.query({
          orgName,
          repoName,
          permission
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    checkOrgPermission: async (orgName, permission) => {
      try {
        return await this.trpc.access.checkOrgPermission.query({
          orgName,
          permission
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  diagnostics = {
    ping: async () => {
      const response = await this.fetchWithAuth(`${this.apiUrl.replace(/\/$/, "")}/health`);
      if (!response.ok) {
        throw new WarmHubError("BACKEND", `Health request failed with status ${response.status}`, response.status);
      }
      return await response.json();
    },
    capabilities: async () => {
      try {
        return await this.trpc.diagnostics.capabilities.query();
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  component = {
    list: async (orgName, repoName, opts) => {
      try {
        return await this.trpc.component.list.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          limit: opts?.limit,
          cursor: opts?.cursor
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, componentId, opts) => {
      try {
        return await this.trpc.component.get.query({
          orgName,
          repoName,
          componentId,
          workspaceName: opts?.workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  commit = {
    apply: async (orgName, repoName, message, operations, opts) => {
      try {
        return await submitCommitViaIngest(this, {
          orgName,
          repoName,
          committer: opts?.committer,
          message,
          workspaceName: opts?.workspaceName,
          componentId: opts?.componentId,
          chunkSize: opts?.chunkSize,
          skipExisting: opts?.skipExisting,
          operations: operations.map((operation) => {
            const commitOperation = toCommitOperation(operation);
            if (opts?.skipExisting === true && commitOperation.operation === "add") {
              return { ...commitOperation, skipExisting: true };
            }
            return commitOperation;
          })
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    log: async (orgName, repoName, opts) => {
      try {
        return await this.trpc.commit.log.query({
          orgName,
          repoName,
          limit: opts?.limit,
          cursor: opts?.cursor,
          committer: opts?.committer,
          opFilter: opts?.opFilter,
          search: opts?.search,
          workspaceName: opts?.workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    logBatch: async (orgName, repoNames, opts) => {
      try {
        return await this.trpc.commit.logBatch.query({
          orgName,
          repoNames,
          limit: opts?.limit
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, commitId, opts) => {
      try {
        return await this.trpc.commit.get.query({
          orgName,
          repoName,
          commitId,
          workspaceName: opts?.workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  org = {
    get: async (orgName) => {
      try {
        return await this.trpc.org.get.query({ orgName });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getCallerRole: async (orgName) => {
      try {
        return await this.trpc.org.getCallerRole.query({ orgName });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    list: async (opts) => {
      try {
        const items = await this.trpc.org.list.query({
          includeArchived: opts?.includeArchived
        });
        return { items };
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    create: async (name, displayName, description) => {
      try {
        return await this.trpc.org.create.mutate({
          name,
          displayName,
          description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    setDescription: async (orgName, description) => {
      try {
        return await this.trpc.org.setDescription.mutate({
          orgName,
          description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    rename: async (orgName, newName) => {
      try {
        return await this.trpc.org.rename.mutate({
          orgName,
          newName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    addMember: async (orgName, email, role = "editor") => {
      try {
        return await this.trpc.org.addMember.mutate({
          orgName,
          email,
          role
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    removeMember: async (orgName, email) => {
      try {
        await this.trpc.org.removeMember.mutate({
          orgName,
          email
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    changeMemberRole: async (orgName, email, role) => {
      try {
        return await this.trpc.org.changeMemberRole.mutate({
          orgName,
          email,
          role
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    setMemberScopes: async (orgName, email, scopes) => {
      try {
        await this.trpc.org.setMemberScopes.mutate({
          orgName,
          email,
          scopes
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    clearMemberScopes: async (orgName, email) => {
      try {
        await this.trpc.org.clearMemberScopes.mutate({
          orgName,
          email
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    listMembers: async (orgName, opts) => {
      try {
        return await this.trpc.org.listMembers.query({
          orgName,
          pending: opts?.pending
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    archive: async (orgName) => {
      try {
        return await this.trpc.org.archive.mutate({
          orgName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    unarchive: async (orgName) => {
      try {
        return await this.trpc.org.unarchive.mutate({
          orgName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  repo = {
    get: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.get.query({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getStats: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.stats.query({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getStatsBatch: async (orgName, repoNames) => {
      try {
        return await this.trpc.repo.statsBatch.query({
          orgName,
          repoNames
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getConfigureStats: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.configureStats.query({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getShapeInstanceCounts: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.shapeInstanceCounts.query({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    list: async (orgName, opts) => {
      try {
        return await this.trpc.repo.list.query({
          orgName,
          limit: opts?.limit,
          cursor: opts?.cursor,
          includeArchived: opts?.includeArchived
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    create: async (orgName, repoName, description, visibility) => {
      try {
        return await this.trpc.repo.create.mutate({
          orgName,
          repoName,
          description,
          visibility
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    setDescription: async (orgName, repoName, description) => {
      try {
        return await this.trpc.repo.setDescription.mutate({
          orgName,
          repoName,
          description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    setVisibility: async (orgName, repoName, visibility) => {
      try {
        return await this.trpc.repo.setVisibility.mutate({
          orgName,
          repoName,
          visibility
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    rename: async (orgName, repoName, newName) => {
      try {
        return await this.trpc.repo.rename.mutate({
          orgName,
          repoName,
          newName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    archive: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.archive.mutate({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    unarchive: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.unarchive.mutate({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    delete: async (orgName, repoName) => {
      try {
        const result = await this.trpc.repo.delete.mutate({
          orgName,
          repoName
        });
        return { graceExpiresAt: new Date(result.graceExpiresAt) };
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    hardDelete: async (orgName, repoName) => {
      try {
        await this.trpc.repo.hardDelete.mutate({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    listPage: async (orgName, opts) => {
      try {
        return await this.trpc.repo.listPage.query({
          orgName,
          limit: opts?.limit,
          cursor: opts?.cursor,
          includeArchived: opts?.includeArchived
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getReadme: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.getReadme.query({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    generateReadme: async (orgName, repoName) => {
      try {
        return await this.trpc.repo.generateReadme.mutate({
          orgName,
          repoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  shape = {
    list: async (orgName, repoName, opts) => {
      try {
        return await this.trpc.shape.list.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          match: opts?.match,
          componentId: opts?.componentId,
          excludeComponents: opts?.excludeComponents
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, shapeName, opts) => {
      try {
        return await this.trpc.shape.get.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          shapeName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    create: async (orgName, repoName, shapeName, fields, opts) => {
      try {
        return await this.trpc.shape.create.mutate({
          orgName,
          repoName,
          shapeName,
          fields,
          description: opts?.description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    revise: async (orgName, repoName, shapeName, newFields, opts) => {
      try {
        return await this.trpc.shape.revise.mutate({
          orgName,
          repoName,
          shapeName,
          newFields,
          description: opts?.description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    remove: async (orgName, repoName, shapeName) => {
      try {
        return await this.trpc.shape.remove.mutate({
          orgName,
          repoName,
          shapeName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    rename: async (orgName, repoName, oldName, newName) => {
      try {
        return await this.trpc.shape.rename.mutate({
          orgName,
          repoName,
          oldName,
          newName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  subscription = {
    create: async (input) => {
      try {
        const result = await this.trpc.subscription.create.mutate(sanitizeSubscriptionCreateInput(input));
        return normalizeSubscriptionInfo(result);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, name) => {
      try {
        const result = await this.trpc.subscription.get.query({
          orgName,
          repoName,
          name
        });
        return normalizeSubscriptionInfo(result);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    list: async (orgName, repoName) => {
      try {
        const result = await this.trpc.subscription.list.query({
          orgName,
          repoName
        });
        return normalizeSubscriptionList(result);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    update: async (input) => {
      try {
        const result = await this.trpc.subscription.update.mutate(sanitizeSubscriptionUpdateInput(input));
        return normalizeSubscriptionInfo(result);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    pause: async (orgName, repoName, name) => {
      try {
        return await this.trpc.subscription.pause.mutate({
          orgName,
          repoName,
          name
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    resume: async (orgName, repoName, name) => {
      try {
        return await this.trpc.subscription.resume.mutate({
          orgName,
          repoName,
          name
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    remove: async (orgName, repoName, name) => {
      try {
        return await this.trpc.subscription.remove.mutate({
          orgName,
          repoName,
          name
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    bindCredentials: async (orgName, repoName, subscriptionName, credentialSetName) => {
      try {
        return await this.trpc.subscription.bindCredentials.mutate({
          orgName,
          repoName,
          subscriptionName,
          credentialSetName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    unbindCredentials: async (orgName, repoName, subscriptionName) => {
      try {
        return await this.trpc.subscription.unbindCredentials.mutate({
          orgName,
          repoName,
          subscriptionName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  action = {
    acquireLease: async (orgName, repoName, subscriptionName, holderId, holderType, opts) => {
      try {
        return await this.trpc.action.acquireLease.mutate({
          orgName,
          repoName,
          subscriptionName,
          holderId,
          holderType,
          graceMs: opts?.graceMs,
          ttlMs: opts?.ttlMs
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    heartbeatLease: async (orgName, repoName, subscriptionName, holderId, ttlMs) => {
      try {
        return await this.trpc.action.heartbeatLease.mutate({
          orgName,
          repoName,
          subscriptionName,
          holderId,
          ttlMs
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    releaseLease: async (orgName, repoName, subscriptionName, holderId) => {
      try {
        return await this.trpc.action.releaseLease.mutate({
          orgName,
          repoName,
          subscriptionName,
          holderId
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    claimDelivery: async (orgName, repoName, subscriptionName, commitId, holderId) => {
      try {
        return await this.trpc.action.claimDelivery.mutate({
          orgName,
          repoName,
          subscriptionName,
          commitId,
          holderId
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    completeDelivery: async (orgName, repoName, subscriptionName, commitId, holderId) => {
      try {
        return await this.trpc.action.completeDelivery.mutate({
          orgName,
          repoName,
          subscriptionName,
          commitId,
          holderId
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    liveFeed: async (orgName, repoName, subscriptionName, opts) => {
      try {
        return await this.trpc.action.liveFeed.query({
          orgName,
          repoName,
          subscriptionName,
          limit: opts?.limit,
          cursor: opts?.cursor
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    listRuns: async (orgName, repoName, opts) => {
      try {
        return await this.trpc.action.listRuns.query({
          orgName,
          repoName,
          subscriptionName: opts?.subscriptionName,
          status: opts?.status,
          since: opts?.since,
          limit: opts?.limit
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getRunAttempts: async (orgName, repoName, subscriptionName, commitId) => {
      try {
        return await this.trpc.action.getRunAttempts.query({
          orgName,
          repoName,
          subscriptionName,
          commitId
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    listNotifications: async (orgName, repoName, opts) => {
      try {
        return await this.trpc.action.listNotifications.query({
          orgName,
          repoName,
          since: opts?.since,
          limit: opts?.limit
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  thing = {
    head: async (orgName, repoName, opts) => {
      try {
        const kind = opts?.kind === "shape" || opts?.kind === "thing" || opts?.kind === "assertion" || opts?.kind === "collection" ? opts.kind : undefined;
        return await this.trpc.thing.head.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          shape: opts?.shape,
          kind,
          match: opts?.match,
          limit: opts?.limit,
          cursor: opts?.cursor,
          componentId: opts?.componentId,
          excludeComponents: opts?.excludeComponents
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, wref, version, opts) => {
      try {
        return await this.trpc.thing.get.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          wref,
          version,
          includeInactive: opts?.includeInactive
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    graph: async (orgName, repoName, wref, opts) => {
      try {
        return await this.trpc.thing.graph.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          wref,
          version: opts?.version,
          depth: opts?.depth,
          limit: opts?.limit
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getMany: async (orgName, repoName, wrefs, version, opts) => {
      try {
        return await this.trpc.thing.getMany.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          wrefs,
          version
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    history: async (orgName, repoName, opts) => {
      try {
        return await this.trpc.thing.history.query({
          orgName,
          repoName,
          workspaceName: opts.workspaceName,
          wref: opts.wref,
          limit: opts.limit,
          cursor: opts.cursor
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    rename: async (orgName, repoName, shapeName, oldName, newName) => {
      try {
        return await this.trpc.thing.rename.mutate({
          orgName,
          repoName,
          shapeName,
          oldName,
          newName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    resolve: async (orgName, repoName, wref, opts) => {
      try {
        return await this.trpc.thing.resolve.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          wref
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    about: async (orgName, repoName, wref, opts) => {
      try {
        return await this.trpc.thing.about.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          wref,
          shape: opts?.shape,
          match: opts?.match,
          depth: opts?.depth,
          resolveCollections: opts?.resolveCollections,
          limit: opts?.limit,
          cursor: opts?.cursor
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    query: async (orgName, repoName, opts) => {
      try {
        const kind = opts?.kind === "shape" || opts?.kind === "thing" || opts?.kind === "assertion" || opts?.kind === "collection" ? opts.kind : undefined;
        return await this.trpc.thing.query.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          shape: opts?.shape,
          about: opts?.about,
          kind,
          match: opts?.match,
          includeInactive: opts?.includeInactive,
          resolveCollections: opts?.resolveCollections,
          limit: opts?.limit,
          cursor: opts?.cursor,
          componentId: opts?.componentId,
          excludeComponents: opts?.excludeComponents,
          excludeInfraShapes: opts?.excludeInfraShapes
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    search: async (orgName, repoName, query, opts) => {
      try {
        const kind = opts?.kind === "shape" || opts?.kind === "thing" || opts?.kind === "assertion" || opts?.kind === "collection" ? opts.kind : undefined;
        return await this.trpc.thing.search.query({
          orgName,
          repoName,
          query,
          workspaceName: opts?.workspaceName,
          shape: opts?.shape,
          about: opts?.about,
          kind,
          match: opts?.match,
          includeInactive: opts?.includeInactive,
          resolveCollections: opts?.resolveCollections,
          limit: opts?.limit,
          cursor: opts?.cursor,
          componentId: opts?.componentId,
          excludeComponents: opts?.excludeComponents,
          excludeInfraShapes: opts?.excludeInfraShapes,
          mode: opts?.mode
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    count: async (orgName, repoName, opts) => {
      try {
        const kind = opts?.kind === "shape" || opts?.kind === "thing" || opts?.kind === "assertion" || opts?.kind === "collection" ? opts.kind : undefined;
        return await this.trpc.thing.count.query({
          orgName,
          repoName,
          workspaceName: opts?.workspaceName,
          shape: opts?.shape,
          about: opts?.about,
          kind,
          match: opts?.match,
          includeInactive: opts?.includeInactive,
          resolveCollections: opts?.resolveCollections,
          componentId: opts?.componentId,
          excludeComponents: opts?.excludeComponents,
          excludeInfraShapes: opts?.excludeInfraShapes
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    refs: async (orgName, repoName, wref, opts) => {
      try {
        return await this.trpc.thing.refs.query({
          orgName,
          repoName,
          wref,
          direction: opts?.direction,
          fieldPath: opts?.fieldPath,
          limit: opts?.limit,
          cursor: opts?.cursor,
          workspaceName: opts?.workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  live = {
    thingHead: async (orgName, repoName, opts, onUpdate) => {
      return this.watchRepoQuery(orgName, repoName, opts?.signal, () => this.thing.head(orgName, repoName, {
        workspaceName: opts?.workspaceName,
        shape: opts?.shape,
        kind: opts?.kind,
        match: opts?.match,
        limit: opts?.limit,
        cursor: opts?.cursor
      }), onUpdate);
    },
    thingHistory: async (orgName, repoName, opts, onUpdate) => {
      return this.watchRepoQuery(orgName, repoName, opts.signal, () => this.thing.history(orgName, repoName, {
        wref: opts.wref,
        workspaceName: opts.workspaceName,
        limit: opts.limit,
        cursor: opts.cursor
      }), onUpdate);
    },
    subscriptionLog: async (orgName, repoName, subscriptionName, opts, onUpdate) => {
      return this.watchRepoQuery(orgName, repoName, opts?.signal, () => this.action.liveFeed(orgName, repoName, subscriptionName, {
        limit: opts?.limit,
        cursor: opts?.cursor
      }), onUpdate);
    },
    subscribe: async (orgName, repoName, opts, onEvent) => {
      const controller = new AbortController;
      const signal = linkAbortSignal(opts?.signal, controller);
      let updates = 0;
      const closed = (async () => {
        const ticket = await this.trpc.live.issueTicket.mutate({
          orgName,
          repoName
        });
        const stream = await this.openSse(ticket.ticket, signal, async (event) => {
          if (event.type !== "invalidate") {
            return;
          }
          const data = event.data;
          updates++;
          await onEvent({
            topic: data.topic,
            commitId: data.commitId,
            affectedShapes: data.affectedShapes,
            affectedThings: data.affectedThings,
            affectedTargets: data.affectedTargets,
            hasNewCommit: data.hasNewCommit
          });
        });
        await stream.ready;
        if (signal.aborted) {
          return { updates };
        }
        await stream.closed;
        return { updates };
      })();
      return {
        close() {
          controller.abort();
        },
        closed
      };
    }
  };
  workspace = {
    create: async (orgName, repoName, name, description) => {
      try {
        return await this.trpc.workspace.create.mutate({
          orgName,
          repoName,
          name,
          description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    list: async (orgName, repoName, opts) => {
      try {
        const result = await this.trpc.workspace.list.query({
          orgName,
          repoName,
          status: opts?.status,
          includeDefault: opts?.includeDefault
        });
        return Array.isArray(result) ? result : result.items;
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, workspaceName) => {
      try {
        return await this.trpc.workspace.get.query({
          orgName,
          repoName,
          workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    remove: async (orgName, repoName, workspaceName) => {
      try {
        return await this.trpc.workspace.remove.mutate({
          orgName,
          repoName,
          workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    revert: async (orgName, repoName, workspaceName, wref) => {
      try {
        return await this.trpc.workspace.revert.mutate({
          orgName,
          repoName,
          workspaceName,
          wref
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    discard: async (orgName, repoName, workspaceName, wref) => {
      try {
        return await this.trpc.workspace.discard.mutate({
          orgName,
          repoName,
          workspaceName,
          wref
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    status: async (orgName, repoName, workspaceName) => {
      try {
        return await this.trpc.workspace.status.query({
          orgName,
          repoName,
          workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    validate: async (orgName, repoName, workspaceName) => {
      try {
        return await this.trpc.workspace.validate.query({
          orgName,
          repoName,
          workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    rebase: async (orgName, repoName, workspaceName) => {
      try {
        return await this.trpc.workspace.rebase.mutate({
          orgName,
          repoName,
          workspaceName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  token = {
    create: async (input) => {
      try {
        return await this.trpc.token.create.mutate(input);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    list: async () => {
      try {
        return await this.trpc.token.list.query();
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (name) => {
      try {
        return await this.trpc.token.get.query({ name });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    revoke: async (name) => {
      try {
        return await this.trpc.token.revoke.mutate({ name });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  ingest = {
    begin: async (input) => {
      try {
        return await this.trpc.ingest.begin.mutate(input);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    append: async (input) => {
      try {
        return await this.trpc.ingest.append.mutate(input);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    finalize: async (input) => {
      try {
        return await this.trpc.ingest.finalize.mutate(input);
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    status: async (input) => {
      try {
        return await this.trpc.ingest.status.query(input);
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  credential = {
    createSet: async (orgName, repoName, name, opts) => {
      try {
        return await this.trpc.credential.createSet.mutate({
          orgName,
          repoName,
          name,
          scope: opts?.scope ?? "repo",
          description: opts?.description
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    listSets: async (orgName, repoName) => {
      try {
        return await this.trpc.credential.listSets.query({ orgName, repoName });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    getSet: async (orgName, repoName, name) => {
      try {
        return await this.trpc.credential.getSet.query({
          orgName,
          repoName,
          name
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    setKey: async (orgName, repoName, setName, keyName, value) => {
      try {
        return await this.trpc.credential.setKey.mutate({
          orgName,
          repoName,
          setName,
          keyName,
          value
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    setKeys: async (orgName, repoName, setName, secrets) => {
      try {
        return await this.trpc.credential.setKeys.mutate({
          orgName,
          repoName,
          setName,
          secrets
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    unsetKey: async (orgName, repoName, setName, keyName) => {
      try {
        return await this.trpc.credential.unsetKey.mutate({
          orgName,
          repoName,
          setName,
          keyName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    deleteSet: async (orgName, repoName, setName) => {
      try {
        return await this.trpc.credential.deleteSet.mutate({
          orgName,
          repoName,
          setName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    grantRepo: async (orgName, repoName, setName, targetRepoName) => {
      try {
        return await this.trpc.credential.grantRepo.mutate({
          orgName,
          repoName,
          setName,
          targetRepoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    ungrantRepo: async (orgName, repoName, setName, targetRepoName) => {
      try {
        return await this.trpc.credential.ungrantRepo.mutate({
          orgName,
          repoName,
          setName,
          targetRepoName
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    listAuditLog: async (orgName, repoName, setName, opts) => {
      try {
        return await this.trpc.credential.listAuditLog.query({
          orgName,
          repoName,
          setName,
          limit: opts?.limit
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    revokeSet: async (orgName, repoName, setName, opts) => {
      try {
        return await this.trpc.credential.revokeSet.mutate({
          orgName,
          repoName,
          setName,
          reason: opts?.reason
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  credentials = this.credential;
  constructor(apiUrlOrOptions, maybeOptions) {
    const options = typeof apiUrlOrOptions === "string" ? maybeOptions : apiUrlOrOptions;
    const apiUrl = typeof apiUrlOrOptions === "string" ? apiUrlOrOptions : apiUrlOrOptions?.apiUrl ?? DEFAULT_API_URL;
    this.apiUrl = apiUrl;
    this.fetchImpl = options?.fetch;
    this.accessToken = options?.accessToken ?? options?.auth?.getToken;
    this.functionLogMode = options?.functionLogs ?? "off";
    if (typeof this.accessToken === "function") {
      const provider = this.accessToken;
      this.getToken = async () => await provider();
    } else if (this.accessToken) {
      const token = this.accessToken;
      this.getToken = async () => token;
    }
    const url = `${apiUrl.replace(/\/$/, "")}/trpc`;
    const fetch2 = (input, init) => this.fetchWithAuth(input, init);
    Object.defineProperty(this, "trpc", {
      value: createTRPCClient({
        links: [
          splitLink({
            condition: (op) => UNBATCHED_TRPC_PATHS.has(op.path),
            true: httpLink({ url, fetch: fetch2 }),
            false: httpBatchLink({ url, fetch: fetch2 })
          })
        ]
      }),
      enumerable: false,
      writable: false,
      configurable: true
    });
  }
  withAccessToken(accessToken) {
    return new _WarmHubClient({
      apiUrl: this.apiUrl,
      fetch: this.fetchImpl,
      accessToken
    });
  }
  actions = this.action;
  changeset = {
    publish: async (orgName, repoName, workspaceName, message, rationale, opts) => {
      try {
        return await this.trpc.changeset.publish.mutate({
          orgName,
          repoName,
          workspaceName,
          message,
          rationale,
          squash: opts?.squash
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    list: async (orgName, repoName, opts) => {
      try {
        const items = await this.trpc.changeset.list.query({
          orgName,
          repoName,
          status: opts?.status,
          limit: opts?.limit
        });
        return { items };
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    get: async (orgName, repoName, changesetNumber) => {
      try {
        return await this.trpc.changeset.get.query({
          orgName,
          repoName,
          changesetNumber
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    accept: async (orgName, repoName, changesetNumber) => {
      try {
        return await this.trpc.changeset.accept.mutate({
          orgName,
          repoName,
          changesetNumber
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    reject: async (orgName, repoName, changesetNumber, reason) => {
      try {
        return await this.trpc.changeset.reject.mutate({
          orgName,
          repoName,
          changesetNumber,
          reason
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    withdraw: async (orgName, repoName, changesetNumber) => {
      try {
        return await this.trpc.changeset.withdraw.mutate({
          orgName,
          repoName,
          changesetNumber
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    },
    validate: async (orgName, repoName, changesetNumber) => {
      try {
        return await this.trpc.changeset.validate.query({
          orgName,
          repoName,
          changesetNumber
        });
      } catch (error) {
        throw toWarmHubError(error);
      }
    }
  };
  async openSse(ticket, signal, onEvent) {
    const response = await this.fetchWithAuth(`${this.apiUrl.replace(/\/$/, "")}/sse?ticket=${encodeURIComponent(ticket)}`, {
      headers: {
        accept: "text/event-stream"
      },
      signal
    });
    if (!response.ok) {
      throw new WarmHubError("BACKEND", `SSE request failed with status ${response.status}`, response.status);
    }
    const reader = response.body?.getReader();
    if (!reader) {
      throw new WarmHubError("BACKEND", "SSE response body is missing");
    }
    let resolveReady = () => {};
    const ready = new Promise((resolve) => {
      resolveReady = resolve;
    });
    const closed = (async () => {
      const decoder = new TextDecoder;
      let buffer = "";
      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) {
            break;
          }
          buffer += decoder.decode(value, { stream: true }).replaceAll(`\r
`, `
`);
          while (true) {
            const separatorIndex = buffer.indexOf(`

`);
            if (separatorIndex === -1) {
              break;
            }
            const chunk = buffer.slice(0, separatorIndex);
            buffer = buffer.slice(separatorIndex + 2);
            const parsed = parseSseChunk(chunk);
            if (!parsed) {
              continue;
            }
            if (parsed.type === "ready") {
              resolveReady();
              continue;
            }
            await onEvent(parsed);
          }
        }
      } catch (error) {
        if (isAbortError2(error) || signal.aborted) {
          return;
        }
        throw error;
      } finally {
        resolveReady();
        reader.releaseLock();
      }
    })();
    return {
      ready,
      closed
    };
  }
  async fetchWithAuth(input, init) {
    const fetchImpl = this.fetchImpl ?? globalThis.fetch;
    const headers = new Headers(init?.headers);
    const token = await resolveAccessToken(this.accessToken);
    if (token) {
      headers.set("authorization", `Bearer ${token}`);
    }
    const doFetch = async (reqInit) => {
      try {
        return await fetchImpl(input, reqInit);
      } catch (err) {
        if (isConnectionError(err)) {
          throw new WarmHubError("NETWORK", connectionErrorMessage(this.apiUrl));
        }
        throw err;
      }
    };
    const response = await doFetch({ ...init, headers });
    if (response.status === 401 && typeof this.accessToken === "function") {
      const freshToken = await resolveAccessToken(this.accessToken);
      if (freshToken && freshToken !== token) {
        const retryHeaders = new Headers(init?.headers);
        retryHeaders.set("authorization", `Bearer ${freshToken}`);
        return doFetch({ ...init, headers: retryHeaders });
      }
    }
    return response;
  }
  async watchRepoQuery(orgName, repoName, signalInput, query, onUpdate) {
    const controller = new AbortController;
    const signal = linkAbortSignal(signalInput, controller);
    let updates = 0;
    const closed = (async () => {
      const emit = async () => {
        const result = await query();
        updates += 1;
        await onUpdate(result);
      };
      const ticket = await this.trpc.live.issueTicket.mutate({
        orgName,
        repoName
      });
      const stream = await this.openSse(ticket.ticket, signal, async (event) => {
        if (event.type !== "invalidate") {
          return;
        }
        await emit();
      });
      await stream.ready;
      if (signal.aborted) {
        return { updates };
      }
      await emit();
      await stream.closed;
      return { updates };
    })();
    return {
      close() {
        controller.abort();
      },
      closed
    };
  }
};
async function resolveAccessToken(provider) {
  if (!provider) {
    return;
  }
  if (typeof provider === "string") {
    return provider;
  }
  return provider();
}
function linkAbortSignal(external, controller) {
  if (!external) {
    return controller.signal;
  }
  if (external.aborted) {
    controller.abort();
    return controller.signal;
  }
  external.addEventListener("abort", () => controller.abort(), { once: true });
  return controller.signal;
}
function parseSseChunk(chunk) {
  let event = "message";
  const dataLines = [];
  for (const line of chunk.split(`
`)) {
    if (!line || line.startsWith(":")) {
      continue;
    }
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim();
      continue;
    }
    if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trim());
    }
  }
  if (dataLines.length === 0) {
    return null;
  }
  const joined = dataLines.join(`
`);
  return {
    type: event,
    data: JSON.parse(joined)
  };
}
function isAbortError2(error) {
  return error instanceof Error && error.name === "AbortError";
}
function toCommitOperation(operation) {
  if (operation.operation === "retract") {
    return {
      operation: "retract",
      name: operation.name,
      ...operation.kind ? { kind: operation.kind } : {},
      ...operation.reason ? { reason: operation.reason } : {}
    };
  }
  if (operation.operation === "revise") {
    const name = operation.name ?? operation.wref;
    if (!name) {
      throw new Error("revise operation requires a target");
    }
    const kind2 = operation.kind ?? inferKind(name, operation);
    if (Object.hasOwn(operation, "active")) {
      throw new Error(`${kind2} revise operation no longer supports 'active' \u2014 use retract('${name}') instead`);
    }
    if (operation.data === undefined) {
      throw new Error(`${kind2} revise operation requires 'data'`);
    }
    if (kind2 === "collection") {
      throw new Error(`collection revise is no longer supported \u2014 use retract('${name}') instead`);
    }
    if (kind2 === "assertion") {
      return {
        operation: "revise",
        kind: "assertion",
        name,
        data: operation.data
      };
    }
    return {
      operation: "revise",
      kind: kind2,
      name,
      data: operation.data
    };
  }
  const kind = operation.kind ?? inferKind(operation.name, operation);
  const skipExisting = "skipExisting" in operation ? operation.skipExisting : undefined;
  if (kind === "collection") {
    if (!("members" in operation) || !operation.members) {
      throw new Error("collection add operation requires 'members'");
    }
    return {
      operation: "add",
      kind: "collection",
      name: operation.name,
      type: "type" in operation && operation.type || "set",
      members: operation.members,
      ...skipExisting === true ? { skipExisting } : {}
    };
  }
  if (!operation.name) {
    throw new Error("add operation requires a name");
  }
  if (kind === "assertion") {
    if (!("about" in operation) || operation.about === undefined) {
      throw new Error("assertion add operation requires 'about'");
    }
    if (operation.data === undefined) {
      throw new Error("assertion add operation requires 'data'");
    }
    return {
      operation: "add",
      kind: "assertion",
      name: operation.name,
      about: operation.about,
      data: operation.data,
      ...skipExisting === true ? { skipExisting } : {}
    };
  }
  if (operation.data === undefined) {
    throw new Error(`${kind} add operation requires 'data'`);
  }
  return {
    operation: "add",
    kind,
    name: operation.name,
    data: operation.data,
    ...skipExisting === true ? { skipExisting } : {}
  };
}
function inferKind(name, operation) {
  if (operation.kind) {
    return operation.kind;
  }
  if ("about" in operation && operation.about !== undefined) {
    return "assertion";
  }
  if ("type" in operation && operation.type !== undefined || "members" in operation && operation.members !== undefined) {
    return "collection";
  }
  if (!name) {
    return "shape";
  }
  return name.includes("/") ? "thing" : "shape";
}
// src/warmhub.ts
var DEFAULT_HOME_REPO = "fish-kill-attribution/chesapeake-attribution";
function clientFromEnv() {
  const apiUrl = process.env.WARMHUB_API_URL ?? "https://api.warmhub.ai";
  const token = process.env.WH_TOKEN ?? process.env.WARMHUB_TOKEN;
  if (!token)
    throw new Error("WH_TOKEN not set (sprite runtime should inject)");
  return new WarmHubClient({ apiUrl, accessToken: () => token });
}
function homeRepo() {
  return process.env.WARMHUB_HOME_REPO ?? process.env.WARMHUB_REPO ?? DEFAULT_HOME_REPO;
}
function splitRepo(repo) {
  const [orgName, repoName] = repo.split("/");
  if (!orgName || !repoName) {
    throw new Error(`Expected repo in "org/name" form, got ${JSON.stringify(repo)}`);
  }
  return { orgName, repoName };
}
async function loadCredentialsFromPayload(payload) {
  const token = payload?.secretToken ?? payload?.payload?.secretToken;
  if (!token)
    return;
  const apiUrl = process.env.WARMHUB_API_URL ?? "https://api.warmhub.ai";
  const resp = await fetch(`${apiUrl}/api/credentials/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token })
  });
  if (!resp.ok) {
    throw new Error(`credentials/export ${resp.status}: ${await resp.text()}`);
  }
  const { secrets } = await resp.json();
  for (const [k, v] of Object.entries(secrets ?? {})) {
    if (process.env[k] === undefined)
      process.env[k] = v;
  }
}

// src/action.ts
function resolveModel() {
  return process.env.PERSONA_MODEL ?? "anthropic/claude-3-haiku";
}
function resolveOpenRouterBase() {
  return process.env.OPENROUTER_BASE ?? "https://openrouter.ai/api/v1";
}
var CONTEXT_LIMIT = 40;
var RADIUS_DEG = 0.75;
var DAYS_BEFORE = 14;
var CAUSES = ["agricultural", "thermal", "industrial", "stormflow", "biological", "unknown"];
var UPSTREAM_REPOS = {
  noaa: "fish-kill-attribution/noaa-sst-daily",
  usgs: "fish-kill-attribution/usgs-nwis",
  epa: "fish-kill-attribution/epa-tri",
  reports: "fish-kill-attribution/state-fishkills"
};
var PERSONA_PRIORS = {
  ngo: `Your priors favor causes linked to industrial pollution, agricultural runoff, stormwater discharge, and regulatory failures. You weigh EPA TRI releases and nutrient loading heavily.`,
  industry: `Your priors favor natural and climatic causes \u2014 harmful algal blooms, thermal stress, hypoxia, fish disease, and stormflow. You are skeptical of facility attribution absent strong contemporaneous exceedances.`,
  agency: `Your priors are balanced and method-conservative. You weigh regulatory compliance history, environmental conditions, and prior adjudicated cases equally.`,
  academic: `Your priors favor methodologically-validated causation. You weight multiple independent data streams and quantitative consistency.`
};
function clamp01(n) {
  if (!Number.isFinite(n))
    return 0;
  return Math.max(0, Math.min(1, n));
}
function normalizeSl(b) {
  const belief = clamp01(b.sl_belief);
  const disbelief = clamp01(b.sl_disbelief);
  const uncertainty = clamp01(b.sl_uncertainty);
  const sum = belief + disbelief + uncertainty;
  if (sum === 0)
    return { ...b, sl_belief: 0, sl_disbelief: 0, sl_uncertainty: 1 };
  return {
    ...b,
    sl_belief: belief / sum,
    sl_disbelief: disbelief / sum,
    sl_uncertainty: uncertainty / sum,
    sl_base_rate: clamp01(b.sl_base_rate ?? 0.5),
    share: clamp01(b.share)
  };
}
function toDateMs(iso) {
  if (!iso)
    return 0;
  const t = Date.parse(iso);
  return Number.isFinite(t) ? t : 0;
}
function degDistance(a, b) {
  return Math.sqrt((a.lat - b.lat) ** 2 + (a.lon - b.lon) ** 2);
}
function qualifyWref(repo, shape, name) {
  return `wh:${repo}/${shape}/${name}`;
}
function normalizeSourceRepo(raw) {
  if (typeof raw === "string")
    return raw;
  if (raw && typeof raw === "object") {
    const r = raw;
    const org = r.org ?? r.orgName ?? r.organization;
    const repo = r.repo ?? r.repoName ?? r.repository;
    if (org && repo)
      return `${org}/${repo}`;
  }
  return "";
}
function revisedObservationWref(payload) {
  const sourceRepo = normalizeSourceRepo(payload.sourceRepo) || "fish-kill-attribution/noaa-sst-daily";
  const ops = payload.matchedOperations ?? [];
  for (const rawOp of ops) {
    const op = rawOp?.operation ?? rawOp;
    if (op.operation !== "revise")
      continue;
    const name = op.name ?? op.wref ?? "";
    if (!name)
      continue;
    if (name.startsWith("wh:"))
      return name;
    const qualified = name.includes("/") ? name : `Observation/${name}`;
    return `wh:${sourceRepo}/${qualified}`;
  }
  return null;
}
async function fetchAllBeliefs(client, org, repo) {
  const all = [];
  let cursor;
  for (let i = 0;i < 20; i++) {
    const page = await client.thing.query(org, repo, {
      shape: "AttributionBelief",
      kind: "assertion",
      limit: 500,
      cursor
    });
    const items = page.items ?? [];
    all.push(...items);
    cursor = page.nextCursor ?? page.cursor;
    if (!cursor || items.length === 0)
      break;
  }
  return all;
}
async function queryNearby(client, repo, shape, event) {
  const { orgName, repoName } = splitRepo(repo);
  let result;
  try {
    result = await client.thing.query(orgName, repoName, { shape, limit: CONTEXT_LIMIT * 3 });
  } catch (err) {
    console.error(`query ${repo}/${shape} failed:`, err.message);
    return [];
  }
  const eventMs = toDateMs(event.date);
  const windowMs = DAYS_BEFORE * 86400000;
  const picks = [];
  for (const row of result.items ?? []) {
    const data = row.data ?? row.head?.data ?? {};
    const lat = Number(data.lat);
    const lon = Number(data.lon);
    if (Number.isFinite(lat) && Number.isFinite(lon)) {
      if (degDistance({ lat, lon }, event) > RADIUS_DEG)
        continue;
    }
    const ts = data.timestamp ?? data.date ?? null;
    if (ts) {
      const tMs = toDateMs(ts);
      if (tMs && eventMs && (tMs > eventMs || eventMs - tMs > windowMs))
        continue;
    }
    picks.push({
      wref: qualifyWref(repo, shape, row.name),
      summary: JSON.stringify({ name: row.name, ...Object.fromEntries(Object.entries(data).slice(0, 10)) })
    });
    if (picks.length >= CONTEXT_LIMIT)
      break;
  }
  return picks;
}
async function gatherContext(client, event) {
  const [o1, o2, r, s] = await Promise.all([
    queryNearby(client, UPSTREAM_REPOS.noaa, "Observation", event),
    queryNearby(client, UPSTREAM_REPOS.usgs, "Observation", event),
    queryNearby(client, UPSTREAM_REPOS.epa, "Release", event),
    queryNearby(client, UPSTREAM_REPOS.reports, "FishKillReport", event)
  ]);
  return [...o1, ...o2, ...r, ...s];
}
function systemPrompt(persona) {
  return `You are an attribution analyst with ${persona} priors.
${PERSONA_PRIORS[persona]}

Given a fish-kill event and nearby upstream observations, releases, and prior
reports, produce causal attribution beliefs across these causes:
${CAUSES.join(", ")}.

For each cause you assign non-zero share, produce a Subjective Logic opinion
(sl_belief + sl_disbelief + sl_uncertainty = 1.0) and cite concrete evidence
wrefs drawn from the provided context.

Output STRICT JSON:
{ "beliefs": [ { "cause": "...", "share": 0..1, "rationale": "...", "sl_belief": 0..1, "sl_disbelief": 0..1, "sl_uncertainty": 0..1, "sl_base_rate": 0..1, "evidence_ids": ["..."] } ] }

Only include causes whose share > 0. Shares should sum to ~1.0. Do not invent wrefs.`;
}
async function askLlm(apiKey, persona, event, context) {
  const resp = await fetch(`${resolveOpenRouterBase()}/chat/completions`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json",
      "HTTP-Referer": "https://github.com/ntoft/correction-reactor",
      "X-Title": "fish-kill-attribution-correction-reactor"
    },
    body: JSON.stringify({
      model: resolveModel(),
      response_format: { type: "json_object" },
      messages: [
        { role: "system", content: systemPrompt(persona) },
        { role: "user", content: JSON.stringify({ event, context: context.map((c) => ({ wref: c.wref, data: c.summary })) }, null, 2) }
      ]
    })
  });
  if (!resp.ok)
    throw new Error(`OpenRouter ${resp.status}: ${await resp.text()}`);
  const res = await resp.json();
  const raw = res.choices?.[0]?.message?.content ?? "{}";
  const parsed = JSON.parse(raw);
  const beliefs = Array.isArray(parsed.beliefs) ? parsed.beliefs : [];
  return beliefs.filter((b) => CAUSES.includes(b.cause)).map(normalizeSl).filter((b) => b.share > 0);
}
async function main() {
  const client = clientFromEnv();
  const { orgName, repoName } = splitRepo(homeRepo());
  const raw = await Bun.stdin.text();
  const rawPayload = raw ? JSON.parse(raw) : {};
  await loadCredentialsFromPayload(rawPayload);
  const payload = rawPayload?.payload ?? rawPayload;
  const revisedWref = revisedObservationWref(payload);
  if (!revisedWref) {
    console.log(JSON.stringify({ skipped: true, reason: "no revise op in payload" }));
    return;
  }
  const allBeliefs = await fetchAllBeliefs(client, orgName, repoName);
  const affected = allBeliefs.filter((b) => {
    const ev = b.data?.evidence_ids ?? b.head?.data?.evidence_ids ?? [];
    return Array.isArray(ev) && ev.some((id) => id === revisedWref || id.endsWith(revisedWref));
  });
  if (affected.length === 0) {
    console.log(JSON.stringify({ revisedWref, affected: 0, note: "no beliefs cite this observation" }));
    return;
  }
  const groups = new Map;
  for (const b of affected) {
    const data = b.data ?? b.head?.data ?? {};
    const about = b.aboutWref ?? b.about ?? b.head?.aboutWref ?? b.head?.about ?? "";
    const persona = data.persona ?? "";
    if (!about || !persona)
      continue;
    const aboutBase = about.replace(/@v\d+$/, "");
    const key = `${aboutBase}::${persona}`;
    let g = groups.get(key);
    if (!g) {
      g = { eventWref: aboutBase, persona, beliefs: [] };
      groups.set(key, g);
    }
    g.beliefs.push({ name: b.name, originalData: data });
  }
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey)
    throw new Error("OPENROUTER_API_KEY not set (credential binding?)");
  const summary = [];
  for (const group of groups.values()) {
    const eventThing = await client.thing.get(orgName, repoName, group.eventWref);
    const eventData = eventThing.head?.data ?? eventThing.data ?? {};
    if (!eventData?.date || eventData.lat == null) {
      summary.push({ eventWref: group.eventWref, persona: group.persona, status: "event missing data" });
      continue;
    }
    const context = await gatherContext(client, eventData);
    const newBeliefs = await askLlm(apiKey, group.persona, eventData, context);
    const byCause = new Map;
    for (const nb of newBeliefs)
      byCause.set(nb.cause, nb);
    const ops = [];
    for (const { name, originalData } of group.beliefs) {
      const m = name.match(/-(agricultural|thermal|industrial|stormflow|biological|unknown)$/);
      const cause = m?.[1] ?? originalData.cause ?? "unknown";
      const nb = byCause.get(cause);
      const reviseName = name.startsWith("AttributionBelief/") ? name : `AttributionBelief/${name}`;
      const identity2 = {
        cause,
        persona: group.persona,
        prompt_version: originalData.prompt_version ?? "reactor-v1"
      };
      if (!nb) {
        const revise2 = {
          operation: "revise",
          kind: "assertion",
          name: reviseName,
          data: {
            ...identity2,
            share: 0,
            sl_belief: 0,
            sl_disbelief: 1,
            sl_uncertainty: 0,
            sl_base_rate: originalData.sl_base_rate ?? 0.5,
            rationale: "Revised: upstream evidence changed; this cause no longer supported by persona.",
            evidence_ids: originalData.evidence_ids ?? [],
            model: originalData.model ?? resolveModel()
          }
        };
        ops.push(revise2);
        continue;
      }
      const revise = {
        operation: "revise",
        kind: "assertion",
        name: reviseName,
        data: {
          ...identity2,
          share: nb.share,
          sl_belief: nb.sl_belief,
          sl_disbelief: nb.sl_disbelief,
          sl_uncertainty: nb.sl_uncertainty,
          sl_base_rate: nb.sl_base_rate ?? 0.5,
          rationale: nb.rationale,
          evidence_ids: nb.evidence_ids ?? [],
          model: resolveModel()
        }
      };
      ops.push(revise);
    }
    const result = await client.commit.apply(orgName, repoName, `correction-reactor: revised ${group.persona} beliefs for ${group.eventWref} (trigger=${revisedWref})`, ops);
    summary.push({
      eventWref: group.eventWref,
      persona: group.persona,
      commitId: result.commitId,
      revisedBeliefs: ops.length
    });
  }
  console.log(JSON.stringify({ revisedWref, affectedBeliefs: affected.length, groups: groups.size, results: summary }));
}
main().catch((err) => {
  console.error(err);
  process.exit(1);
});
