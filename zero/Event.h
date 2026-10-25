#pragma once

#include <utility>
#include <vector>

namespace zero {

template <typename T>
struct EventTypeDispatcher;

// Event scopes, for many games in one process (the duel simulator, bot-ai ticket 45): a handler belongs to the scope
// current when it was made, and an event goes to the handlers of the current scope and to those of scope 0, which
// hear every scope. A bot alone in its process never sets one, so everything is scope 0 as before.
inline unsigned int g_EventScope = 0;

struct Event {
  virtual ~Event() {}

  template <typename T>
  static void Dispatch(const T& event) {
    EventTypeDispatcher<T>::Get().Dispatch(event);
  }
};

// This is an event handler for a specific type of event.
// Users of this should inherit from each type of EventHandler<T> that they want to process.
// Implement void HandleEvent(const T& event) to handle the dispatched event.
template <typename T>
struct EventHandler {
 protected:
  EventHandler() : event_scope(g_EventScope) { EventTypeDispatcher<T>::Get().RegisterHandler(this); }

 public:
  unsigned int event_scope;

  virtual ~EventHandler() { EventTypeDispatcher<T>::Get().UnregisterHandler(this); }

  virtual void HandleEvent(const T& event) = 0;
};

// This class stores the handlers for the given Event type T.
template <typename T>
struct EventTypeDispatcher {
 private:
  std::vector<EventHandler<T>*> handlers;

  virtual ~EventTypeDispatcher() {}

 public:
  static EventTypeDispatcher& Get() {
    static EventTypeDispatcher instance;
    return instance;
  }

  void Dispatch(const T& event) {
    unsigned int scope = g_EventScope;
    for (auto& handler : handlers) {
      if (handler->event_scope == 0 || handler->event_scope == scope) handler->HandleEvent(event);
    }
  }

  void RegisterHandler(EventHandler<T>* handler) { handlers.push_back(handler); }

  void UnregisterHandler(EventHandler<T>* handler) {
    for (size_t i = 0; i < handlers.size(); ++i) {
      auto check = handlers[i];
      if (check == handler) {
        std::swap(handlers[i], handlers[handlers.size() - 1]);
        handlers.pop_back();
        break;
      }
    }
  }
};

}  // namespace zero
