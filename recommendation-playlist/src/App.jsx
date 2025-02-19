import { useEffect, useState } from "react"
import axios from "axios"

export default function App() {
  const [users, setUsers] = useState([])
  const [selectedUser, setSelectedUser] = useState("")
  const [recommendations, setRecommendations] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const colors = [
    "red-200",
    "blue-200",
    "green-200",
    "yellow-200",
    "purple-200",
    "pink-200",
    "indigo-200",
    "teal-200",
    "orange-200",
    "gray-200"
  ]
  useEffect(() => {
    fetch("http://127.0.0.1:8000/users")
      .then((res) => res.json())
      .then((data) => setUsers(data.users || []))
      .catch((error) => console.error("Lỗi tải users:", error))
  }, [])

  const fetchRecommendations = () => {
    if (!selectedUser) return

    setIsLoading(true)
    axios
      .get(`http://127.0.0.1:8000/recommend/${selectedUser}`)
      .then((response) => setRecommendations(response.data.recommendations || []))
      .catch((error) => {
        console.error("Lỗi lấy gợi ý:", error)
        setRecommendations([])
      })
      .finally(() => setIsLoading(false))
  }

  return (
    <div className="p-4 max-w-lg mx-auto">
      <h1 className="text-xl font-bold mb-4">🎵 Gợi ý nhạc</h1>

      {/* Dropdown chọn user */}
      <div className="mb-4">
        <label className="block mb-2">Chọn người dùng:</label>
        <select
          className="p-2 border rounded w-full"
          value={selectedUser}
          onChange={(e) => setSelectedUser(e.target.value)}
        >
          <option value="">-- Chọn một người dùng --</option>
          {users.map((user) => (
            <option key={user} value={user}>
              {user}
            </option>
          ))}
        </select>
      </div>

      {/* Nút lấy gợi ý */}
      <button
        className="px-4 py-2 bg-blue-500 text-white rounded disabled:opacity-50"
        onClick={fetchRecommendations}
        disabled={!selectedUser || isLoading}
      >
        {isLoading ? "Đang tải..." : "Gợi ý nhạc"}
      </button>

      {/* Danh sách bài hát gợi ý */}
      <ul className="mt-4">
        {recommendations.length > 0 ? (
          recommendations.map((track, index) => {
            const randomColor = colors[Math.floor(Math.random() * colors.length)];
            return (
              <li key={index} className={`flex items-center space-x-2 font-semibold p-2 border-b bg-${randomColor}`}>
                <p>{index + 1}.</p> <p className="capitalize">{track}</p>
              </li>
            )
          })
        ) : (
          <p className="text-gray-500 mt-2">
            {isLoading ? "Đang lấy dữ liệu..." : "Không có bài hát nào được đề xuất."}
          </p>
        )}
      </ul>
    </div>
  )
}