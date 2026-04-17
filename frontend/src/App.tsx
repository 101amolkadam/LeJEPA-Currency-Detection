import {
  createRouter,
  RouterProvider,
  createRootRoute,
  createRoute,
  Outlet,
} from "@tanstack/react-router";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/lib/queryClient";
import HomePage from "@/pages/HomePage";
import ResultsPage from "@/pages/ResultsPage";
import HistoryPage from "@/pages/HistoryPage";

const rootRoute = createRootRoute({
  component: Outlet,
  notFoundComponent: () => (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="text-4xl mb-4">🔍</div>
        <h1 className="text-xl font-bold text-gray-700">Page Not Found</h1>
        <p className="text-gray-500 mt-2">The page you're looking for doesn't exist.</p>
        <a href="/" className="mt-4 inline-block bg-blue-600 text-white px-6 py-2 rounded-lg">
          Go Home
        </a>
      </div>
    </div>
  ),
  errorComponent: ({ error }) => (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="text-4xl mb-4">❌</div>
        <h1 className="text-xl font-bold text-red-600">Something went wrong</h1>
        <p className="text-gray-600 mt-2">{String(error)}</p>
        <a href="/" className="mt-4 inline-block bg-blue-600 text-white px-6 py-2 rounded-lg">
          Go Home
        </a>
      </div>
    </div>
  ),
});

const indexRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: HomePage,
});

const resultsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/results/$id",
  component: ResultsPage,
});

const historyRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/history",
  component: HistoryPage,
});

const routeTree = rootRoute.addChildren([indexRoute, resultsRoute, historyRoute]);

const router = createRouter({ routeTree });

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  );
}
